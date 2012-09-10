import time

class TimeCounter(object):
    def __init__(self, log_func):
        self.log_func = log_func
        self.start = self.last = time.time()

    def step(self):
        """Set a new checkpoint and return the number of seconds since
        the last checkpoint.  If this is the first step, return the time
        since the timer was started.
        """
        now = time.time()
        elapsed = now - self.last
        self.last = now
        return elapsed

    def total(self):
        """Return the number of seconds since the timer was started.
        Does not set a checkpoint.
        """
        return time.time() - self.start

    def _log(self, what, secs):
        msg = "{} in {:0.2f} seconds".format(what, secs)
        self.log_func(msg)

    def finish(self, what="finished"):
        self._log(what, self.total())

    def checkpoint(self, what):
        secs = self.step()
        self._log(what, secs)

    def checkpoint_count(self, seq, whats, action="read"):
        count = len(seq)
        secs = self.step()
        what = "{} {} {}".format(action, count, whats)
        self._log(what, secs)

def test():
    tc = TimeCounter()
