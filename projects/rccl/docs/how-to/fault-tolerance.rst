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

When a communicator is created in non-blocking mode (``config.blocking = 0``),
RCCL calls can return ``ncclInProgress`` before the operation has finished. Poll
:cpp:func:`ncclCommGetAsyncError` until the communicator leaves the
``ncclInProgress`` state, yielding the CPU between checks. The following helper is
the pattern used by the RCCL MPI tests (see ``test/RevokeMPITests.cpp``):

.. code-block:: cpp

   // Poll ncclCommGetAsyncError until the comm leaves ncclInProgress,
   // yielding the CPU between checks. Returns the terminal state.
   static ncclResult_t waitForAsyncResult(ncclComm_t comm)
   {
       ncclResult_t state = ncclInProgress;
       while (state == ncclInProgress)
       {
           ncclResult_t r = ncclCommGetAsyncError(comm, &state);
           if (r != ncclSuccess) return r;
           if (state == ncclInProgress) sched_yield();
       }
       return state;
   }

If the terminal state is anything other than ``ncclSuccess``, the operation has
failed and the communicator must be aborted with :cpp:func:`ncclCommAbort`. Once
work has been enqueued on a stream, use ``hipStreamSynchronize`` to wait for the
device-side collective to complete, as the tests do after each collective.

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
to abort and whether to restart. The simplest recovery path is to abort the
affected communicators on every rank, synchronize out of band, then rebuild and
continue. The following sequence is taken from the ``Grow_ErrorRecoveryRegrow``
test in ``test/GrowMPITests.cpp``:

.. code-block:: cpp

   // Simulate an error: abort the communicators on every rank.
   ncclCommAbort(grownComm);
   grownComm = nullptr;
   if (initialComm) {
       ncclCommAbort(initialComm);
       initialComm = nullptr;
   }

   // Synchronize all ranks out of band before rebuilding.
   MPI_Barrier(MPI_COMM_WORLD);

   // Recover: create a fresh communicator and grow it again.
   buildComm(existing, &initialComm);
   growByOne(initialComm, existing, &grownComm);

Here ``buildComm`` and ``growByOne`` are the application helpers shown in
:ref:`ft-grow`; ``MPI_Barrier`` is the out-of-band synchronization point that
ensures every rank has finished aborting before any rank starts rebuilding.

.. _ft-shrink:

Shrinking a communicator to drop failed ranks
=============================================

:cpp:func:`ncclCommShrink` creates a new communicator by removing specific ranks
from an existing one. This is the recommended way to recover when only a subset
of GPUs or nodes has failed: the surviving ranks form a smaller communicator and
keep running.

Pass a list of ranks to exclude. Only the ranks that will remain in the new
communicator call :cpp:func:`ncclCommShrink`; the excluded ranks must not call
it. RCCL re-orders the remaining ranks to keep the numbering contiguous. The
following example excludes the last rank, following the ``Grow_ThenShrink`` test
in ``test/GrowMPITests.cpp``:

.. code-block:: cpp

   int        excludeRank = worldSize - 1;  // Rank to exclude.
   ncclComm_t shrunkComm  = nullptr;

   if (wr != excludeRank) {
       ncclResult_t res = ncclCommShrink(comm, &excludeRank, 1, &shrunkComm,
                                         nullptr, NCCL_SHRINK_DEFAULT);
       if (res != ncclSuccess) {
           // Handle error.
       }
       // Use shrunkComm for subsequent collectives, then destroy it when done.
       ncclCommDestroy(shrunkComm);
   }

When you shrink **after** a failure, there may be operations still in flight on
the parent communicator. Use the ``NCCL_SHRINK_ABORT`` flag so RCCL terminates
those operations first, then builds the smaller communicator. This mirrors the
``ShrinkAbort_InFlight_ChildWorks_RankRenumbering`` test:

.. code-block:: cpp

   if (wr != excludeRank) {
       // NCCL_SHRINK_ABORT aborts in-progress work on the parent before shrinking.
       ncclResult_t res = ncclCommShrink(comm, &excludeRank, 1, &shrunkComm,
                                         nullptr, NCCL_SHRINK_ABORT);
       ncclCommUserRank(shrunkComm, &childRank);   // ranks are renumbered
       ncclCommCount(shrunkComm, &childSize);      // contiguous in the child
   }

The two shrink flags are:

* ``NCCL_SHRINK_DEFAULT`` -- shrink a parent communicator that has no
  outstanding operations, or one whose in-flight work was already aborted by
  :cpp:func:`ncclCommRevoke` (see :ref:`ft-revoke`).
* ``NCCL_SHRINK_ABORT`` -- first abort ongoing parent operations, then shrink.
  Use this for fault-tolerance recovery when you shrink directly, without a
  preceding revoke, and the parent may still have collectives in flight.

.. _ft-grow:

Growing a communicator to restore capacity
==========================================

:cpp:func:`ncclCommGrow` creates a new communicator by adding ranks to an
existing one. After a shrink has removed failed ranks, grow lets you bring
replacement GPUs or nodes back in and return the job to full size.

Growing requires coordination between the existing ranks and the new ranks. A
coordinator rank from the existing communicator generates a unique ID with
:cpp:func:`ncclCommGetUniqueId` and distributes it to the new ranks through an
out-of-band channel (the RCCL tests use ``MPI_Bcast``). The parameter usage is:

* Existing ranks: ``comm`` set, ``rank = -1``. The coordinator (rank 0) passes
  ``uniqueId = &growId``; other existing ranks may pass ``uniqueId = NULL``.
* New ranks: ``comm = NULL``, ``uniqueId = &growId``, ``rank =`` the assigned rank.

The following helper grows a communicator by one rank, following the
``growByOne`` helper in ``test/GrowMPITests.cpp``:

.. code-block:: cpp

   ncclResult_t growByOne(ncclComm_t existingComm, int existingNRanks,
                          ncclComm_t* outComm)
   {
       const int wr       = world_rank;          // MPI world rank
       const int newRank  = existingNRanks;      // the single new rank
       const int newTotal = existingNRanks + 1;

       // Coordinator (rank 0) generates the grow ID and broadcasts it.
       ncclUniqueId growId{};
       if (wr == 0) {
           NCCLCHECK(ncclCommGetUniqueId(existingComm, &growId));
       }
       MPI_Bcast(&growId, sizeof(growId), MPI_BYTE, 0, MPI_COMM_WORLD);

       if (wr < existingNRanks) {
           // Existing ranks: comm set, rank = -1.
           NCCLCHECK(ncclCommGrow(existingComm, newTotal, &growId, -1,
                                  outComm, nullptr));
       } else if (wr == newRank) {
           // New rank: comm = NULL, rank = assigned.
           NCCLCHECK(ncclCommGrow(nullptr, newTotal, &growId, newRank,
                                  outComm, nullptr));
       }
       return ncclSuccess;
   }

For a non-blocking grow, pass a config with ``blocking = 0`` and poll for
completion before using the new communicator, as in the ``Grow_NonBlocking``
test:

.. code-block:: cpp

   ncclConfig_t nbConfig = NCCL_CONFIG_INITIALIZER;
   nbConfig.blocking = 0;

   if (wr < existing) {
       ncclCommGrow(initialComm, worldSize, &growId, -1, &grownComm, &nbConfig);
   } else if (wr == existing) {
       ncclCommGrow(nullptr, worldSize, &growId, wr, &grownComm, &nbConfig);
   }

   ncclResult_t asyncErr = ncclInProgress;
   while (asyncErr == ncclInProgress) {
       ncclCommGetAsyncError(grownComm, &asyncErr);
   }
   // asyncErr == ncclSuccess once the grow has completed.

After the grow succeeds, you can destroy the parent communicator and keep using
the grown one (see the ``Grow_ParentDestroyAfterGrow`` test):

.. code-block:: cpp

   ncclCommDestroy(initialComm);   // parent no longer needed
   // ... collectives on grownComm ...
   ncclCommDestroy(grownComm);     // when finished

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
the output buffers of an aborted collective contain undefined data.

The typical recovery flow is *collective -> revoke -> shrink -> collective*, as
exercised by the ``Collective_Revoke_Shrink_Collective`` test in
``test/RevokeMPITests.cpp``:

.. code-block:: cpp

   // A collective is in flight on the parent when a peer fails.
   ncclAllReduce(send_buf, recv_buf, count, ncclFloat, ncclSum, parent, stream);

   // Revoke aborts the in-flight collective but keeps the parent usable.
   ncclCommRevoke(parent, NCCL_REVOKE_DEFAULT);

   MPI_Barrier(MPI_COMM_WORLD);

   // Build a smaller communicator from the survivors. Because revoke already
   // aborted the in-flight work, NCCL_SHRINK_DEFAULT is used here (not ABORT).
   ncclComm_t child = NCCL_COMM_NULL;
   if (!isExcluded) {
       ncclCommShrink(parent, excludeList.data(), excludeList.size(),
                      &child, nullptr, NCCL_SHRINK_DEFAULT);
   }

   MPI_Barrier(MPI_COMM_WORLD);

   // Continue on the child communicator.
   if (!isExcluded) {
       ncclAllReduce(send_buf, recv_buf, count, ncclFloat, ncclSum, child, stream);
       hipStreamSynchronize(stream);
   }

The RCCL revoke tests establish the following contract:

* After a revoke, any newly enqueued collective on the revoked communicator
  returns ``ncclInvalidUsage`` (``Revoke_RejectsCollectives``).
* A revoked communicator is still valid as a parent for
  :cpp:func:`ncclCommSplit` and :cpp:func:`ncclCommShrink`
  (``Revoke_ThenSplit_ChildWorks``, ``RevokeThenShrink_ChildWorks``).
* A revoked communicator can be torn down cleanly with
  :cpp:func:`ncclCommDestroy` (``Revoke_ThenDestroy_CleanLifecycle``).
* Revoking the same communicator twice is rejected with ``ncclInvalidArgument``
  (``Revoke_DoubleRevoke_Rejected``).
* Calling :cpp:func:`ncclCommFinalize` on a revoked communicator is rejected with
  ``ncclInvalidUsage`` -- use :cpp:func:`ncclCommDestroy` instead
  (``Revoke_ThenFinalize_Rejected``).

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
* When only some ranks fail, prefer shrinking over a full job restart. Either
  shrink directly with ``NCCL_SHRINK_ABORT`` to terminate in-flight work, or call
  :cpp:func:`ncclCommRevoke` first and then shrink with ``NCCL_SHRINK_DEFAULT``.
* Use the ROCm tooling (``amd-smi``, ``rocminfo``, ``ibv_devinfo``) and RAS logs
  to diagnose the root cause before re-creating communicators.

Worked examples
===============

The RCCL MPI test suite is the authoritative source of working examples for
these APIs. See:

* ``test/GrowMPITests.cpp`` -- grow, non-blocking grow, grow/shrink elastic
  cycles, and abort/recreate error recovery.
* ``test/RevokeMPITests.cpp`` -- revoke, revoke -> split, revoke -> shrink,
  ``NCCL_SHRINK_ABORT``, and non-blocking revoke.

Related links
=============

* :ref:`troubleshooting-rccl`
* :ref:`api-library`
* :ref:`env-variables`
