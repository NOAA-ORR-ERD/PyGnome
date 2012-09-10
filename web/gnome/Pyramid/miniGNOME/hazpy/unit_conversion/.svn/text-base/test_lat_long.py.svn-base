#!/usr/bin/env python

import lat_long
import unittest

class testSignBit(unittest.TestCase):
    
    def testNeg(self):
        self.assertTrue(lat_long.signbit(-5.0))
    def testPos(self):
        self.assertFalse(lat_long.signbit(5.0))
    def testNegZero(self):
        self.assertTrue(lat_long.signbit(-0.0))
    def testPosZero(self):
        self.assertFalse(lat_long.signbit(0.0))
    def testIntegerNeg(self):
        self.assertTrue(lat_long.signbit(-5))
    def testIntegerPos(self):
        self.assertFalse(lat_long.signbit(5))


class testLatLongErrors(unittest.TestCase):
    LLC = lat_long.LatLongConverter

    def testNegativeMinutes(self):
        self.assertRaises(ValueError, self.LLC.ToDecDeg, 30, -30)

    def testNegativeSeconds(self):
        self.assertRaises(ValueError, self.LLC.ToDecDeg, 30, 30, -1)

    def testTooBig(self):
        self.assertRaises(ValueError , self.LLC.ToDecDeg, d=200)

    def testTooNegative(self):
        self.assertRaises(ValueError , self.LLC.ToDecDeg, d=-181)

    def testTooBigMin(self):
        self.assertRaises(ValueError , self.LLC.ToDecDeg, d=20, m=61)

    def testTooBigSec(self):
        self.assertRaises(ValueError , self.LLC.ToDecDeg, d=30, m=42, s=61)

    def testDegFractAndMin(self):
        self.assertRaises(ValueError , self.LLC.ToDecDeg, d=30.2, m=5, s=0)

    def testDegFractAndSec(self):
        self.assertRaises(ValueError , self.LLC.ToDecDeg, d=30.2, m=0, s=6.3)
        
    def testMinFractAndSec(self):
        self.assertRaises(ValueError , self.LLC.ToDecDeg, d=30, m=4.5, s=6)
        

class testLatLong(unittest.TestCase):
    LLC = lat_long.LatLongConverter

    def testDecDegrees(self):
        self.assertEqual(self.LLC.ToDecDeg(30, 30), 30.5)

    def testDecDegrees2(self):
        self.assertAlmostEqual(self.LLC.ToDecDeg(30, 30, 30), 30.50833333333)

    def testDegMin(self):
        self.assertEqual(self.LLC.ToDegMin(30.5)[0], 30 )
        self.assertAlmostEqual(self.LLC.ToDegMin(30.5)[1], 30.0 )
   
    def testMinusZeroDeg(self):
        self.assertEqual(self.LLC.ToDecDeg(d=-0.0, m=20, s=20), -0.33888888888888885)

    def testBinaryProblem(self):
        self.assertEqual(self.LLC.ToDegMinSec(45.05), (45, 3, 0.0))
        
    def testDDString(self):
        d, m, s = 120, 30, 5
        DecDeg = self.LLC.ToDecDeg(d, m, s, ustring=True)
        self.assertEqual(self.LLC.ToDecDeg(d, m, s, ustring=True),
                         u"120.501389\xb0")

    def testDDString2(self):
        d, m, s = -50, 30, 5
        DecDeg = self.LLC.ToDecDeg(d, m, s, ustring=True)
        self.assertEqual(self.LLC.ToDecDeg(d, m, s, ustring=True),
                         u"-50.501389\xb0")

    def testDDString2(self):
        d, m, s = 0, 30, 0
        DecDeg = self.LLC.ToDecDeg(d, m, s, ustring=True)
        self.assertEqual(self.LLC.ToDecDeg(d, m, s, ustring=True),
                         u"0.500000\xb0")

    def testDMString(self):
        d, m = 120, 45.5
        DecDeg = self.LLC.ToDecDeg(d, m)
        self.assertEqual(self.LLC.ToDegMin(DecDeg, True),
                         u"120\xb0 45.500'")

    def testDMString2(self):
        d, m = -120, 3
        DecDeg = self.LLC.ToDecDeg(d, m)
        self.assertEqual(self.LLC.ToDegMin(DecDeg, True),
                         u"-120\xb0 3.000'")

    def testDMSString(self):
        d, m, s = 120, 45,  15
        DecDeg = self.LLC.ToDecDeg(d, m, s)
        self.assertEqual(self.LLC.ToDegMinSec(DecDeg, True),
                         u"120\xb0 45' 15.00\"")

    def testDMSString2(self):
        d, m, s = -120, 3,  15
        DecDeg = self.LLC.ToDecDeg(d, m, s)
        self.assertEqual(self.LLC.ToDegMinSec(DecDeg, True),
                         u"-120\xb0 3' 15.00\"")

    def testDMtringZero(self):
        d, m, s = -0.0, 3,  0
        DecDeg = self.LLC.ToDecDeg(d, m, s)
        self.assertEqual(self.LLC.ToDegMin(DecDeg, True),
                         u"""-0\xb0 3.000'""")

    def testDMSStringZero(self):
        d, m, s = -0.0, 3,  15
        DecDeg = self.LLC.ToDecDeg(d, m, s)
        self.assertEqual(self.LLC.ToDegMinSec(DecDeg, True),
                         u'''-0\xb0 3' 15.00"''')



class testLatitude(unittest.TestCase):
    L = lat_long.Latitude

    def testTooBig(self):
        self.assertRaises(ValueError , self.L, deg=95)

    def testTooNeg(self):
        self.assertRaises(ValueError , self.L, deg=-95)

    def testSignAndDir(self):
        self.assertRaises(ValueError , self.L, deg=-45, direction="N")

    def testNegativeSeconds(self):
        self.assertRaises(ValueError, self.L, deg=0.0, min=0.0, sec=-0.001)

    def testNegativeMinutes(self):
        self.assertRaises(ValueError, self.L, deg=0.0, min=-0.1)

    def testDirS(self):
        self.assertEqual(self.L(deg=45, direction="S").value, -45.0)
        
    def testDirN(self):
        self.assertEqual(self.L(deg=45.5, direction="N").value, 45.5)

    def testDirWrong(self):
        self.assertRaises(ValueError, self.L, deg=45, direction="Whatever")
        
    def testDM1(self):
        self.assertEqual(self.L(deg=45.5, direction="S").degrees_minutes(), (45, 30.0, 'South'))

    def testDMS1(self):
        self.assertEqual(self.L(deg=45.5, direction="S").degrees_minutes_seconds(), (45, 30.0, 0, 'South'))

    def testDMS2(self):
        self.assertEqual(self.L(deg=30.001).degrees_minutes_seconds(), (30, 0, 3.6, 'North'))
        
    def testFormat1(self):
        self.assertEqual(self.L(deg=30.001).format(1), u'30.00\xb0 North')
        
    def testFormat2(self):
        self.assertEqual(self.L(deg=30.001).format(2), u"30\xb0 0.06' North")
        
    def testFormat3(self):
        self.assertEqual(self.L(deg=30.001).format(3), u'''30\xb0 0' 3.60" North''')
        
    def testFormat4(self):
        self.assertEqual(self.L(deg=-30.001).format(1), u'30.00\xb0 South')
        
    def testFormatHTML3(self):
        self.assertEqual(self.L(deg=30.001).format_html(3), '30&deg; 0\' 3.60" North')

    def testDirectionN(self):
        self.assertEqual(self.L(deg=80.5).direction(), 'North')

    def testDirectionS(self):
        self.assertEqual(self.L(deg=-80.5).direction(), 'South')

    def testDirectionZero(self):
        # note: should this be North or South???
        self.assertEqual(self.L(deg=0.0).direction(), 'North')

    def testDirectionSmall(self):
        self.assertEqual(self.L(deg=-.00000000000001).direction(), 'South')

    
class testLongitude(unittest.TestCase):
    L = lat_long.Longitude

    def testTooBig(self):
        self.assertRaises(ValueError , self.L, deg=185)

    def testTooNeg(self):
        self.assertRaises(ValueError , self.L, deg=-185)

    def testSignAndDir(self):
        self.assertRaises(ValueError , self.L, deg=-45, direction="W")

    def testDirW(self):
        self.assertEqual(self.L(deg=45, direction="W").value, -45.0)
        
    def testDirE(self):
        self.assertEqual(self.L(deg=45.5, direction="E").value, 45.5)

    def testDirWrong(self):
        self.assertRaises(ValueError, self.L, deg=45, direction="Anything")
        
    def testDM1(self):
        self.assertEqual(self.L(deg=45.5, direction="W").degrees_minutes(), (45, 30.0, 'West'))

    def testDMS1(self):
        self.assertEqual(self.L(deg=45.5, direction="E").degrees_minutes_seconds(), (45, 30.0, 0, 'East'))

    def testDMS2(self):
        self.assertEqual(self.L(deg=30.001).degrees_minutes_seconds(), (30, 0, 3.6, 'East'))
        
    def testFormat1(self):
        self.assertEqual(self.L(deg=30.001).format(1), u'30.00\xb0 East')
        
    def testFormat2(self):
        self.assertEqual(self.L(deg=30.001).format(2), u"30\xb0 0.06' East")
        
    def testFormat3(self):
        self.assertEqual(self.L(deg=30.001).format(3), u'''30\xb0 0' 3.60" East''')
        
    def testFormat4(self):
        self.assertEqual(self.L(deg=-30.001).format(1), u'30.00\xb0 West')
        
    def testFormatHTML3(self):
        self.assertEqual(self.L(deg=30.001).format_html(3), '30&deg; 0\' 3.60" East')

    def testDirectionN(self):
        self.assertEqual(self.L(deg=80.5).direction(), 'East')

    def testDirectionS(self):
        self.assertEqual(self.L(deg=-80.5).direction(), 'West')

    def testDirectionZero(self):
        # note: should this be East or West???
        self.assertEqual(self.L(deg=0.0).direction(), 'East')

    def testDirectionSmall(self):
        self.assertEqual(self.L(deg=-.00000000000001).direction(), 'West')

    





if __name__ == "__main__":
    unittest.main()

