.. meta::
   :description: How RCCL handles errors and supports fault tolerance for multi-GPU and multi-node collective communication on AMD GPUs
   :keywords: RCCL, ROCm, AMD, fault tolerance, error handling, communicator abort, shrink, grow, revoke

.. _fault-tolerance:

***************************
Fault tolerance in RCCL
***************************

Large-scale training and inference jobs that run across many AMD GPUs and nodes
must be able to survive failures such as a network link going down, an ECC error
on a device, a node crash, or a process that exits unexpectedly. RCCL provides a
set of APIs that let an application detect these conditions, release the affected
resources, and continue making progress without restarting the whole job.

This topic explains the error-handling model that RCCL inherits from the NCCL API
and the AMD-specific extensions that RCCL adds on top of it. The APIs described
here are declared in ``rccl.h`` and follow the standard NCCL signatures, so code
that was written for NCCL fault tolerance continues to work with RCCL. Where the
behavior is unique to RCCL or to AMD hardware, it is called out explicitly.

.. note::

   Fault tolerance relies on RCCL communicators being created in non-blocking
   mode. With ``config.blocking = 0``, every RCCL call (except
   :cpp:func:`ncclCommDestroy` and :cpp:func:`ncclCommAbort`) returns
   immediately, which means the application can react to a hang or a failure
   instead of being stuck inside a collective. See :ref:`async-errors` below.

.. _ft-error-codes:

Error handling and communicator abort
=====================================

Every RCCL call returns an ``ncclResult_t`` code. When a call returns a value
other than ``ncclSuccess`` or ``ncclInProgress`` and ``NCCL_DEBUG`` is set to
``WARN``, RCCL prints a human-readable message describing what happened. Setting
``NCCL_DEBUG=INFO`` additionally prints the call stack that led to the error,
which is helpful when filing an issue with AMD (see :ref:`troubleshooting-rccl`).

The following table summarizes the RCCL error codes and how each one should be
handled.

.. list-table::
   :header-rows: 1
   :widths: 22 30 26 22

   * - Error
     - Description
     - Resolution
     - Group behavior
   * - ``ncclSuccess``
     - No error.
     - None.
     - None.
   * - ``ncclUnhandledCudaError``
     - Error during a HIP (ROCm runtime) call.
     - Fix the HIP/ROCm configuration or usage, then abort the communicator.
     - Global.
   * - ``ncclSystemError``
     - Error during a system call, for example a network or shared-memory failure.
     - Fix the system configuration or usage, then abort the communicator.
     - Global.
   * - ``ncclInternalError``
     - A bug inside RCCL.
     - Abort the communicator and report the issue to AMD.
     - Global.
   * - ``ncclInvalidArgument``
     - An argument to a call is invalid, for example a ``NULL`` pointer.
     - Fix the application. The call had no effect and the communicator is still usable.
     - Individual.
   * - ``ncclInvalidUsage``
     - The sequence of RCCL calls is invalid (a dynamic usage error).
     - Fix the application, then abort the communicator.
     - Global.
   * - ``ncclInProgress``
     - The call is still running (non-blocking mode).
     - Poll for completion with :cpp:func:`ncclCommGetAsyncError`.
     - None.

Notes on the table:

* ``ncclUnhandledCudaError`` and ``ncclSystemError`` mean that a call RCCL made
  into an external component (the HIP runtime, the driver, or the network stack)
  failed. The message identifies the component to investigate. On AMD systems,
  use the ROCm tooling described in :ref:`debugging-system-info` (for example
  ``amd-smi``, ``rocminfo``, and ``ibv_devinfo``) to diagnose the underlying
  hardware or fabric problem.
* ``ncclInvalidArgument`` is non-fatal. The communicator keeps working and the
  application can continue as if the call had not happened.
* The errors marked **Global** are fatal for the communicator. To recover, the
  application must call :cpp:func:`ncclCommAbort` on the communicator (and on
  every communicator in the same group) and then re-create it.

.. _async-errors:

Asynchronous errors and error handling
======================================

Some failures, especially network failures, are not reported by the originating
call. They surface later through :cpp:func:`ncclCommGetAsyncError`. An operation
that hits an asynchronous error usually stops making progress and never
completes, so polling for it is the only reliable way to notice that something
has gone wrong.

Instead of calling ``hipStreamSynchronize`` (which can block forever if a
collective is stuck), wait on the stream while also polling for asynchronous
errors. When an error is detected, abort the communicator with
:cpp:func:`ncclCommAbort`.

.. code-block:: cpp

   int rcclStreamSynchronize(hipStream_t stream, ncclComm_t comm) {
     hipError_t hipErr;
     ncclResult_t rcclErr, rcclAsyncErr;
     while (1) {
       hipErr = hipStreamQuery(stream);
       if (hipErr == hipSuccess)
         return 0;

       if (hipErr != hipErrorNotReady) {
         printf("HIP Error : hipStreamQuery returned %d\n", hipErr);
         return 1;
       }

       rcclErr = ncclCommGetAsyncError(comm, &rcclAsyncErr);
       if (rcclErr != ncclSuccess) {
         printf("RCCL Error : ncclCommGetAsyncError returned %d\n", rcclErr);
         return 1;
       }

       if (rcclAsyncErr != ncclSuccess) {
         // An asynchronous error happened. Stop the operation and destroy
         // the communicator.
         rcclErr = ncclCommAbort(comm);
         if (rcclErr != ncclSuccess)
           printf("RCCL Error : ncclCommAbort returned %d\n", rcclErr);
         // The caller can now abort or create a new communicator.
         return 2;
       }

       // Let other threads (including RCCL background threads) use the CPU.
       sched_yield();
     }
   }

.. _ft-recovery:

Recovering from a failure
=========================

RCCL provides several recovery strategies. Which one to use depends on how much
of the job you want to preserve when a fault occurs:

* **Abort and re-create** the whole communicator. The simplest approach: tear
  everything down and rebuild it. Use this for a coarse, job-wide restart.
* **Shrink** the communicator to drop the failed ranks and keep running with the
  survivors. Use this when only some ranks have failed and the workload can
  continue with fewer GPUs.
* **Grow** the communicator to bring replacement ranks back in. Use this to
  return to full capacity after a shrink, or to scale up on demand.
* **Revoke** (RCCL-specific) the communicator to abort in-flight work without
  destroying the communicator, so it can be reused as a parent for a subsequent
  shrink or grow.

To abort communicators safely, the application must create them in non-blocking
mode and make sure no thread is inside an RCCL call when
:cpp:func:`ncclCommAbort` is invoked. With ``config.blocking = 0``, every RCCL
call (except :cpp:func:`ncclCommDestroy` and :cpp:func:`ncclCommAbort`) is
non-blocking, so the abort can be issued at any point during initialization,
communication, or finalization. If a communicator is blocking, a network error
can leave a thread stuck inside an RCCL call indefinitely.

When any rank in a communicator fails, every other rank must call
:cpp:func:`ncclCommAbort` on its own communicator. The application decides when
to abort and whether to restart. The following example initializes and splits a
communicator in non-blocking mode so that it can be aborted at any point:

.. code-block:: cpp

   bool globalFlag;
   bool abortFlag = false;
   ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
   /* Set the communicator as non-blocking. */
   config.blocking = 0;
   CHECK(ncclCommInitRankConfig(&comm, nRanks, id, myRank, &config));
   do {
     CHECK(ncclCommGetAsyncError(comm, &state));
   } while (state == ncclInProgress && checkTimeout() != true);

   if (checkTimeout() == true || state != ncclSuccess) abortFlag = true;

   /* Synchronize abortFlag across all healthy ranks. */
   reportErrorGlobally(abortFlag, &globalFlag);

   if (globalFlag) {
     /* Timeout or init failure: every rank aborts and restarts. */
     ncclCommAbort(comm);
     /* restartNCCL is a user-provided function. It typically cleans up
      * resources and calls ncclCommInitRankConfig() to rebuild communicators. */
     restartNCCL(&comm);
   }

   /* Non-blocking communicator split. */
   CHECK(ncclCommSplit(comm, color, key, &childComm, &config));
   do {
     CHECK(ncclCommGetAsyncError(comm, &state));
   } while (state == ncclInProgress && checkTimeout() != true);

   if (checkTimeout() == true || state != ncclSuccess) abortFlag = true;

   reportErrorGlobally(abortFlag, &globalFlag);

   if (globalFlag) {
     ncclCommAbort(comm);
     if (childComm != NCCL_COMM_NULL) ncclCommAbort(childComm);
     restartNCCL(&comm);
   }
   /* Application workload. */

The ``checkTimeout`` and ``reportErrorGlobally`` helpers are supplied by the
application. ``checkTimeout`` decides how long to wait for an RCCL operation
before declaring it failed; ``reportErrorGlobally`` propagates the local decision
to the other healthy ranks (for example over MPI or sockets).

.. _ft-shrink:

Shrinking a communicator to drop failed ranks
=============================================

:cpp:func:`ncclCommShrink` creates a new communicator by removing specific ranks
from an existing one. This is the recommended way to recover when only a subset
of GPUs or nodes has failed: the surviving ranks form a smaller communicator and
keep running.

Pass a list of ranks to exclude. Only the ranks that will remain in the new
communicator call :cpp:func:`ncclCommShrink`; the excluded ranks must not call
it. RCCL re-orders the remaining ranks to keep the numbering contiguous.

.. code-block:: cpp

   int excludeRanks[] = {1};  // Rank to exclude.
   int excludeCount = 1;
   ncclComm_t newcomm;

   if (myRank != 1) {
     ncclResult_t res = ncclCommShrink(comm, excludeRanks, excludeCount,
                                       &newcomm, NULL, NCCL_SHRINK_DEFAULT);
     if (res != ncclSuccess) {
       // Handle error.
     }
     // Use newcomm for subsequent collectives, then destroy it when done.
     ncclCommDestroy(newcomm);
   }

When you shrink **after** a failure, there may be operations still in flight on
the parent communicator. Use the ``NCCL_SHRINK_ABORT`` flag so RCCL terminates
those operations first, then builds the smaller communicator:

.. code-block:: cpp

   if (myRank != 1) {
     // NCCL_SHRINK_ABORT aborts in-progress work on the parent before shrinking.
     ncclResult_t res = ncclCommShrink(comm, excludeRanks, excludeCount,
                                       &newcomm, NULL, NCCL_SHRINK_ABORT);
     // ...
   }

The two shrink flags are:

* ``NCCL_SHRINK_DEFAULT`` -- shrink a healthy parent communicator that has no
  outstanding operations.
* ``NCCL_SHRINK_ABORT`` -- first abort ongoing parent operations, then shrink.
  Use this for fault-tolerance recovery, where the parent may be in an
  inconsistent state.

.. _ft-grow:

Growing a communicator to restore capacity
==========================================

:cpp:func:`ncclCommGrow` creates a new communicator by adding ranks to an
existing one. After a shrink has removed failed ranks, grow lets you bring
replacement GPUs or nodes back in and return the job to full size.

Growing requires coordination between the existing ranks and the new ranks. A
coordinator rank from the existing communicator generates a unique ID with
:cpp:func:`ncclCommGetUniqueId` and distributes it to the new ranks through an
out-of-band channel (MPI, sockets, or shared memory). The parameter usage is:

* Existing non-root rank: ``comm`` set, ``uniqueId = NULL``, ``rank = -1``.
* Existing root (coordinator): ``comm`` set, ``uniqueId = &id``, ``rank = -1``.
* New rank: ``comm = NULL``, ``uniqueId = &id``, ``rank =`` the assigned rank.

.. code-block:: cpp

   // Step 1: The coordinator (for example rank 0) generates the grow ID.
   ncclUniqueId growId;
   if (myRank == 0) {
     ncclResult_t res = ncclCommGetUniqueId(comm, &growId);
     if (res != ncclSuccess) {
       // Handle error.
     }
     // Distribute growId to all new ranks out of band (MPI, sockets, ...).
   }

   // Step 2: All existing ranks call ncclCommGrow.
   ncclComm_t newcomm;
   ncclResult_t res = ncclCommGrow(comm, 8, NULL, -1, &newcomm, NULL);

   // Step 3: New ranks call ncclCommGrow with the received growId.
   hipSetDevice(myDevice);
   res = ncclCommGrow(NULL, 8, &growId, myNewRank, &newcomm, NULL);

   // Step 4: For non-blocking grow, poll until the operation completes.
   ncclResult_t asyncErr;
   do {
     res = ncclCommGetAsyncError(newcomm, &asyncErr);
   } while (asyncErr == ncclInProgress);

   // Step 5: Use newcomm for collectives.
   // Step 6: Destroy the parent communicator once grow has succeeded.
   ncclCommDestroy(comm);
   // Step 7: Destroy newcomm when finished.
   ncclCommDestroy(newcomm);

Keep these constraints in mind:

* Existing ranks keep their original rank numbers. New ranks are numbered
  starting from the size of the parent communicator.
* The grow ID is single-use. You cannot generate a new ID while a previous one
  is still unconsumed, and each ID can be consumed by exactly one grow.
* There must be no outstanding operations on the parent communicator when you
  call :cpp:func:`ncclCommGrow`, otherwise the grow can deadlock.
* The new communicator inherits the parent's configuration for existing ranks;
  new ranks use the provided config or the defaults.

.. _ft-revoke:

Revoking a communicator (RCCL extension)
========================================

:cpp:func:`ncclCommRevoke` is an AMD-specific extension that aborts in-flight
collectives **without** destroying the communicator. It raises the
communicator's abort flag, stops the proxy service, and rejects any newly
enqueued collectives with ``ncclInvalidUsage``. Once the asynchronous revoke job
finishes, the abort flag is cleared and the communicator becomes valid again as a
*parent* for :cpp:func:`ncclCommSplit`, :cpp:func:`ncclCommShrink`, or
:cpp:func:`ncclCommGrow`. It can also be torn down with
:cpp:func:`ncclCommDestroy` or :cpp:func:`ncclCommAbort`.

Revoke is intended for recovery scenarios where a peer has failed in the middle
of a collective. Because in-flight collectives are aborted rather than drained,
the output buffers of an aborted collective contain undefined data. This is the
key difference from :cpp:func:`ncclCommFinalize`, which drains outstanding work;
calling :cpp:func:`ncclCommFinalize` on a revoked communicator is invalid and
returns ``ncclInvalidUsage``.

.. code-block:: cpp

   // Abort in-flight work but keep comm usable as a shrink/grow parent.
   ncclResult_t res = ncclCommRevoke(comm, NCCL_REVOKE_DEFAULT);
   if (res != ncclSuccess) {
     // Handle error.
   }

   // After the revoke completes, build a smaller communicator from the survivors.
   ncclComm_t newcomm;
   res = ncclCommShrink(comm, excludeRanks, excludeCount, &newcomm, NULL,
                        NCCL_SHRINK_ABORT);

Pass ``NCCL_REVOKE_DEFAULT`` for ``revokeFlags``; any other value is rejected
with ``ncclInvalidArgument``.

.. _ft-finalize-destroy:

Finalizing and destroying a communicator
========================================

When you want a clean shutdown rather than an abort, finalize the communicator
first. :cpp:func:`ncclCommFinalize` transitions the communicator from
``ncclSuccess`` to ``ncclInProgress``, completes all outstanding operations in
the background, and flushes network-related resources. The communicator returns
to ``ncclSuccess`` once it is globally quiescent; query the state with
:cpp:func:`ncclCommGetAsyncError`. If the communicator is non-blocking, finalize
is non-blocking; otherwise it blocks.

After finalization, free the remaining local resources with
:cpp:func:`ncclCommDestroy`. If the communicator is in the ``ncclSuccess`` state
when :cpp:func:`ncclCommDestroy` is called, the call is guaranteed to be
non-blocking. Do not access the communicator after
:cpp:func:`ncclCommDestroy` returns.

Use :cpp:func:`ncclCommAbort` instead of finalize/destroy when the communicator
is in a bad state and you cannot wait for outstanding operations to drain.

.. _ft-monitoring:

Monitoring health with RAS
==========================

RCCL includes a Reliability, Availability, and Serviceability (RAS) subsystem
that runs a lightweight background network among the processes in a job. It
exchanges keep-alive messages and can report dead peers and per-communicator
status, which makes it easier to identify *which* rank or node failed when a job
stops making progress.

To see RAS activity in the logs, enable the RAS debug subsystem:

.. code-block:: shell

   export NCCL_DEBUG=INFO
   export NCCL_DEBUG_SUBSYS=RAS

Combine this with the diagnostics in :ref:`troubleshooting-rccl` to pinpoint the
failed component before deciding whether to abort, shrink, or grow.

.. _ft-best-practices:

Best practices
==============

* Create communicators with ``config.blocking = 0`` whenever fault tolerance is
  required, so that aborts and timeouts are possible.
* Always pair a wait loop with :cpp:func:`ncclCommGetAsyncError` rather than
  relying on ``hipStreamSynchronize`` alone.
* When one rank decides to abort, propagate that decision to all healthy ranks
  out of band, and have every rank abort its own communicator.
* Prefer :cpp:func:`ncclCommShrink` with ``NCCL_SHRINK_ABORT`` (optionally after
  :cpp:func:`ncclCommRevoke`) over a full job restart when only some ranks fail.
* Use the ROCm tooling (``amd-smi``, ``rocminfo``, ``ibv_devinfo``) and RAS logs
  to diagnose the root cause before re-creating communicators.

Related links
=============

* :ref:`troubleshooting-rccl`
* :ref:`api-library`
* :ref:`env-variables`
