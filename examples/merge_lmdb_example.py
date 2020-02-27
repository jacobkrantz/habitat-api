import os
import random
from typing import List

import lmdb
import msgpack_numpy

LMDB_MAP_SIZE = 1.65e12
MAX_LOAD_SIZE = 500


def save_to(entries: List, db_to: str) -> None:
    inserted = 0
    with lmdb.open(db_to, map_size=int(LMDB_MAP_SIZE)) as lmdb_env:
        txn = lmdb_env.begin(write=True)
        start_id = lmdb_env.stat()["entries"]
        for entry in entries:
            txn.put(
                str(start_id + inserted).encode(),
                msgpack_numpy.packb(entry, use_bin_type=True),
            )
            inserted += 1
        txn.commit()
    return


def load_into(db_from: str, db_to: str, k: int = -1) -> None:
    r"""Loads entries in the lmdb database `db_from` into the database
    `db_to`. Assumes entry keys are indices. If `k` is specified, loads
    that many number of episodes. These episodes are sampled at random
    without replacement from `db_from`. Creates `db_to` if not exist.
    """
    db_to_dir = os.path.dirname(db_to)
    if not os.path.exists(db_to_dir):
        os.makedirs(db_to_dir)

    loaded = []
    with lmdb.open(
        db_from, map_size=int(LMDB_MAP_SIZE), readonly=True, lock=False
    ) as lmdb_env, lmdb_env.begin(buffers=True) as txn:
        if k > -1:
            try:
                eps_to_load = random.sample(
                    list(range(lmdb_env.stat()["entries"])), k=k
                )
            except ValueError as err:
                print(f"Num entries in DB: {lmdb_env.stat()['entries']}")
                raise err
        else:
            eps_to_load = range(lmdb_env.stat()["entries"])

        for i in eps_to_load:
            loaded.append(
                msgpack_numpy.unpackb(txn.get(str(i).encode()), raw=False)
            )
            if len(loaded) == MAX_LOAD_SIZE:
                save_to(loaded, db_to=db_to)
                loaded = []

    if loaded:
        save_to(loaded, db_to=db_to)
    return


if __name__ == "__main__":
    load_into(
        db_from="trajectories_dirs/tf_joint_train_aug/trajectories.lmdb",
        db_to="trajectories_dirs/some_new_db/trajectories.lmdb",
        k=5000,
    )
