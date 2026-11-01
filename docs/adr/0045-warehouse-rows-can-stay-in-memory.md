# Warehouse rows can stay in memory

Status: accepted (issue #734, ALE-497)

Warehouse sources already snapshot rows in memory and assign identities before
extraction. Their `json` backend nevertheless stages each selected row as a
local document file. Some operators cannot place sensitive input in such files.

Add an explicit `json_memory` backend for warehouse sources. It consumes the
same snapshot and projects the same JSON values without a document payload
file. Keep the existing `json` path for projects relying on file-based hooks;
the memory backend rejects those hooks before execution. Switching backends
reprocesses the model, while source document IDs and row fingerprints stay
unchanged.

A warehouse-native root relation could avoid document extraction entirely,
but would need separate source selection, deletion and incremental semantics.
Reusing the existing source contract preserves those semantics. The boundary
does not cover warehouse storage, swap, or downstream output and caches.
