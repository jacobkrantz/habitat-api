import lmdb
import msgpack_numpy

LMDB_MAP_SIZE = 1.65e12
LOAD_SIZE = 500


def save_to(entries, db_to):
    inserted = 0
    with lmdb.open(db_to, map_size=int(LMDB_MAP_SIZE)) as lmdb_env:
        start_id = lmdb_env.stat()["entries"]
        txn = lmdb_env.begin(write=True)
        for entry in entries:
            txn.put(
                str(start_id + inserted).encode(),
                msgpack_numpy.packb(entry, use_bin_type=True),
            )
            inserted += 1
        txn.commit()
    return


def load_into(db_from, db_to):
    r"""Loads all of the entries in the lmdb database `db_from` into the
    database `db_to`. Assumes entry keys are indices. 
    """
    loaded = []
    with lmdb.open(
        db_from, map_size=int(LMDB_MAP_SIZE), readonly=True, lock=False,
    ) as lmdb_env, lmdb_env.begin(buffers=True) as txn:
        for i in range(lmdb_env.stat()["entries"]):
            loaded.append(
                msgpack_numpy.unpackb(txn.get(str(i).encode()), raw=False,)
            )
            if len(loaded) == LOAD_SIZE:
                save_to(loaded, db_to=db_to)
                loaded = []

    if loaded:
        save_to(loaded, db_to=db_to)
    return


if __name__ == "__main__":
    load_into(
        db_from="trajectories_dirs/tf_train/trajectories.lmdb",
        db_to="trajectories_dirs/tf_joint_train_aug/trajectories.lmdb",
    )
