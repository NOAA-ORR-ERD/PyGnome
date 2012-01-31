import argparse
import datetime
import logging
import unittest

import sqlalchemy as sa
import sqlalchemy.orm as orm

import hazpy.sitestats.logger as logger

import hazpy.sitestats.model as model

DBURL = "postgresql:///test"
LOG_SQL = False

class TestLogger(unittest.TestCase):
    def test1(self):
        engine = sa.create_engine(DBURL, echo=LOG_SQL)
        conn = engine.connect()
        model.Base.metadata.drop_all(bind=conn)
        model.Base.metadata.create_all(bind=conn)
        now_str = unicode(datetime.datetime.now().isoformat())
        now_str_lower = now_str.lower()
        url = u"/" + now_str
        query = u"a=1"
        full_url = u"{}?{}".format(url, query)
        referer = u"http://example.com" + full_url
        remote_addr = u"1.1.1.1"
        status = 200
        session = u"ABC"
        user = None
        log = logger.Logger(u"test", conn)
        log.log_access(remote_addr, status, url, session, user)
        #log.log_access_pylons()
        log.log_referer(referer)
        log.log_search(u"test1", now_str, 600)
        log.log_search(u"test1", now_str, 0)
        log.log_search(u"test1", now_str, 0)
        log.log_event(u"test2", now_str)
        sess = orm.Session(bind=engine)
        r = sess.query(model.Access).filter_by(url=url).first()
        self.assertIsNotNone(r)
        self.assertEqual(r.url, url)
        self.assertEqual(r.site, u"test")
        self.assertEqual(r.remote_addr, remote_addr)
        self.assertEqual(r.status, status)
        self.assertEqual(r.session, session)
        self.assertEqual(r.user, user)
        r = sess.query(model.Referer).filter_by(path=url).first()
        self.assertIsNotNone(r)
        self.assertEqual(r.path, url)
        self.assertEqual(r.query, u"a=1")
        success = sess.query(model.Search).filter_by(type=u"test1").one()
        failure = sess.query(model.Search).filter_by(type=u"test1-fail").one()
        self.assertEqual(success.count, 1)
        self.assertEqual(failure.count, 2)
        self.assertEqual(success.value, now_str_lower)
        event = sess.query(model.Event).filter_by(type=u"test2").one()
        self.assertEqual(event.count, 1)
        self.assertEqual(event.value, now_str)


if __name__ == "__main__":  unittest.main()
