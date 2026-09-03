# gnome/outputters/postgis.py

import logging
from datetime import timezone

import numpy as np

from gnome.outputters.outputter import Outputter

log = logging.getLogger(__name__)


class PostGISOutput(Outputter):
    """
    Writes spill element data (position, status, mass) to a PostGIS-enabled
    PostgreSQL database at each model timestep.

    The caller is responsible for providing a ``row_factory`` callable that
    converts a single element dict into whatever row object or dict their
    schema expects, and a ``persist`` callable that receives a list of those
    rows and writes them to the database.

    This keeps the outputter decoupled from any specific ORM, connection pool,
    or application framework.

    Parameters
    ----------
    persist : callable
        ``persist(rows: list) -> None``
        Called once per timestep with all rows for that step.
        Responsible for opening a connection/session, writing, and committing.

    row_factory : callable, optional
        ``row_factory(element: dict) -> any``
        Converts a single element dict (keys: time, lon, lat, depth,
        status_code, mass, step, element_index) into whatever object
        your persistence layer expects (ORM instance, plain dict, tuple…).
        Defaults to returning the element dict unchanged.

    run_id : str, optional
        Arbitrary identifier for this model run (task ID, UUID, etc.).
        Passed into each element dict as ``run_id`` so your row_factory
        can store it. Default is None.

    metadata : dict, optional
        Any additional key/value pairs to attach to every row.
        Merged into each element dict before row_factory is called.

    round_data : bool
        Round float arrays. Default True.

    round_to : int
        Decimal places for rounding. Default 4.
    """

    def __init__(self,
                 persist,
                 row_factory=None,
                 run_id=None,
                 metadata=None,
                 round_data=True,
                 round_to=4,
                 **kwargs):
        super().__init__(**kwargs)

        if not callable(persist):
            raise TypeError("persist must be a callable: persist(rows) -> None")
        if row_factory is not None and not callable(row_factory):
            raise TypeError("row_factory must be a callable: row_factory(element) -> row")

        self._persist = persist
        self._row_factory = row_factory or (lambda x: x)
        self.run_id = run_id
        self.metadata = metadata or {}
        self.round_data = round_data
        self.round_to = round_to

    def _dataarray_p_types(self, data_array):
        if issubclass(data_array.dtype.type, float):
            return data_array.round(self.round_to).astype(float).tolist()
        elif issubclass(data_array.dtype.type, np.integer):
            return data_array.astype(int).tolist()
        else:
            raise TypeError(
                "PostGISOutput can only handle float or integer array types"
            )

    def write_output(self, step_num, islast_step=False):
        super().write_output(step_num, islast_step)

        if not self._write_step:
            return

        tz = timezone(self.timezone_offset.as_timedelta())

        rows = []
        for sc in self.cache.load_timestep(step_num).items():
            positions = self._dataarray_p_types(sc['positions'])
            status_codes = self._dataarray_p_types(sc['status_codes'])
            masses = self._dataarray_p_types(sc['mass'])
            timestamp = sc.current_time_stamp.replace(tzinfo=tz).isoformat()

            for ix, pos in enumerate(positions):
                element = {
                    'run_id': self.run_id,
                    'step': step_num,
                    'element_index': ix,
                    'time': timestamp,
                    'lon': pos[0],
                    'lat': pos[1],
                    'depth': pos[2],
                    'status_code':  status_codes[ix],
                    'mass': masses[ix],
                }
                element.update(self.metadata)

                try:
                    rows.append(self._row_factory(element))
                except Exception:
                    log.exception("row_factory failed for element %d at step %d",
                                  ix, step_num)

        if rows:
            try:
                self._persist(rows)
            except Exception:
                log.exception("persist failed at step %d", step_num)