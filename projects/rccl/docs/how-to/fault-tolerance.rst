.. meta::
   :description: How RCCL handles errors and supports fault tolerance for multi-GPU and multi-node collective communication on AMD GPUs
   :keywords: RCCL, ROCm, AMD, fault tolerance, error handling, communicator abort, shrink, grow, revoke

.. _fault-tolerance:

***************************
Fault tolerance in RCCL
***************************

Large-scale jobs running across many AMD GPUs and nodes must survive failures
such as a network link going down, an ECC error, a node crash, or a process that
exits unexpectedly. RCCL provides APIs to detect these conditions, release the
affected resources, and keep running without restarting the whole job.

RCCL inherits the NCCL error-handling model and adds AMD-specific extensions on
top of it. The APIs are declared in ``rccl.h`` and use the standard NCCL
signatures.

.. note::

   Fault tolerance relies on communicators created in non-blocking mode. With
   ``config.blocking = 0``, every RCCL call (except :cpp:func:`ncclCommDestroy`
   and :cpp:func:`ncclCommAbort`) returns immediately, so the application can
   react to a hang or a failure instead of being stuck inside a collective.

.. _ft-error-codes:

Error handling and communicator abort
=====================================

Every RCCL call returns an ``ncclResult_t`` code. Set ``NCCL_DEBUG=WARN`` to
print a human-readable message on error, or ``NCCL_DEBUG=INFO`` to also print the
call stack.

.. list-table::
   :header-rows: 1
   :widths: 24 34 42

   * - Error
     - Description
     - Handling
   * - ``ncclSuccess``
     - No error.
     - None.
   * - ``ncclUnhandledCudaError``
     - Error during a HIP (ROCm runtime) call.
     - Fatal. Abort the communicator and re-create it.
   * - ``ncclSystemError``
     - Error during a system call, for example a network failure.
     - Fatal. Abort the communicator and re-create it.
   * - ``ncclInternalError``
     - A bug inside RCCL.
     - Fatal. Abort the communicator and report the issue to AMD.
   * - ``ncclInvalidArgument``
     - An argument is invalid, for example a ``NULL`` pointer.
     - Non-fatal. The call had no effect; the communicator is still usable.
   * - ``ncclInvalidUsage``
     - The sequence of RCCL calls is invalid.
     - Fatal. Abort the communicator and re-create it.
   * - ``ncclInProgress``
     - The call is still running (non-blocking mode).
     - Poll with :cpp:func:`ncclCommGetAsyncError`.

A fatal error applies to all communicators in the same group. To recover, call
:cpp:func:`ncclCommAbort` on every affected communicator and re-create it.

.. _async-errors:

Asynchronous errors
===================

Network failures are not reported by the originating call; they surface later
through :cpp:func:`ncclCommGetAsyncError`. In non-blocking mode, poll it until the
communicator leaves the ``ncclInProgress`` state.

.. code-block:: cpp

   ncclResult_t state = ncclInProgress;
   while (state == ncclInProgress) {
       ncclCommGetAsyncError(comm, &state);
   }
   if (state != ncclSuccess) ncclCommAbort(comm);

.. _ft-recovery:

Recovering from a failure
=========================

RCCL offers several recovery strategies, depending on how much of the job you
want to preserve:

* **Abort and re-create** the whole communicator -- a coarse, job-wide restart.
* **Shrink** to drop the failed ranks and keep running with the survivors.
* **Grow** to bring replacement ranks back in and return to full size.
* **Revoke** (RCCL-specific) to abort in-flight work without destroying the
  communicator, so it can be reused as a parent for a later shrink or grow.

Abort requires non-blocking communicators, and no thread may be inside an RCCL
call when :cpp:func:`ncclCommAbort` is invoked.

.. _ft-shrink:

Shrinking a communicator
========================

:cpp:func:`ncclCommShrink` creates a new communicator by removing ranks from an
existing one. Only the ranks that remain call it; the excluded ranks must not.
Remaining ranks are renumbered contiguously.

.. code-block:: cpp

   int excludeRank = nRanks - 1;
   ncclComm_t newComm = nullptr;
   ncclCommShrink(comm, &excludeRank, 1, &newComm, nullptr, NCCL_SHRINK_DEFAULT);

Shrink flags:

* ``NCCL_SHRINK_DEFAULT`` -- the parent has no in-flight work, or it was already
  aborted by :cpp:func:`ncclCommRevoke`.
* ``NCCL_SHRINK_ABORT`` -- abort in-flight work on the parent first, then shrink.
  Use this when shrinking directly after a failure, without a preceding revoke.

.. _ft-grow:

Growing a communicator
======================

:cpp:func:`ncclCommGrow` creates a new communicator by adding ranks. A
coordinator generates a unique ID with :cpp:func:`ncclCommGetUniqueId` and
distributes it to the new ranks out of band.

* Existing ranks: ``comm`` set, ``rank = -1`` (the coordinator passes the ID).
* New ranks: ``comm = NULL``, ``rank =`` the assigned rank, with the ID.

.. code-block:: cpp

   // Existing ranks
   ncclCommGrow(comm, newTotal, &growId, -1, &newComm, nullptr);

   // New ranks
   ncclCommGrow(nullptr, newTotal, &growId, newRank, &newComm, nullptr);

Notes:

* Existing ranks keep their rank numbers; new ranks are numbered from the parent
  size upward.
* The grow ID is single-use.
* The parent must have no outstanding operations when grow is called.

.. _ft-revoke:

Revoking a communicator (RCCL extension)
========================================

:cpp:func:`ncclCommRevoke` aborts in-flight collectives **without** destroying
the communicator, so it can recover from a peer that failed mid-collective.
Output buffers of an aborted collective contain undefined data.

The typical flow is *revoke, then shrink, then continue*. Because revoke already
aborts in-flight work, the shrink uses ``NCCL_SHRINK_DEFAULT``.

.. code-block:: cpp

   ncclCommRevoke(parent, NCCL_REVOKE_DEFAULT);
   ncclCommShrink(parent, excludeList, excludeCount, &child, nullptr, NCCL_SHRINK_DEFAULT);

Behavior:

* New collectives on a revoked communicator return ``ncclInvalidUsage``.
* A revoked communicator stays valid as a parent for :cpp:func:`ncclCommSplit`
  and :cpp:func:`ncclCommShrink`, and can be torn down with
  :cpp:func:`ncclCommDestroy`.
* Revoking twice returns ``ncclInvalidArgument``.
* :cpp:func:`ncclCommFinalize` on a revoked communicator returns
  ``ncclInvalidUsage`` -- use :cpp:func:`ncclCommDestroy` instead.
* ``revokeFlags`` must be ``NCCL_REVOKE_DEFAULT``.

.. _ft-finalize-destroy:

Finalizing and destroying
=========================

For a clean shutdown, call :cpp:func:`ncclCommFinalize` to drain outstanding
operations (the state returns to ``ncclSuccess`` once quiescent), then
:cpp:func:`ncclCommDestroy` to free resources. Do not access a communicator after
it is destroyed. Use :cpp:func:`ncclCommAbort` instead when the communicator is
in a bad state and outstanding operations cannot be drained.
