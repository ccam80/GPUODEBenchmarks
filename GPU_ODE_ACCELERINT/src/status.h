#ifndef STATUS_H
#define STATUS_H

//! Per-trajectory return codes written at exit, one int32 each.
#define STATUS_FILE "status.bin"

//! Allocate the per-trajectory status array and register the end-of-run summary.
void status_init(int NUM);

//! Per-trajectory return code, indexed by accelerInt's return codes.
extern int* status_codes;

#endif
