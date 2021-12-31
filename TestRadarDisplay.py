import datetime as dt
import GPMPy as gpy
import unittest
  
class RadarDisplayTest(unittest.TestCase):

    def setUp(self):
        self.data_path = "./data/"
        self.rad_disp = gpy.RadarDisplay(self.data_path, "sghaemi1@umd.edu", "sghaemi1@umd.edu")
        pass


    # def test_get_files_by_dt(self):        
    #     st = dt.datetime(2021, 9, 1, 9, 0, 0)
    #     ed = dt.datetime(2021, 9, 1, 12, 0, 0)

    #     new_dates = self.rad_disp.get_files_by_dt(st, ed)
    #     self.assertTrue(new_dates == self.rad_disp.files)
    
    def test_combine(self):
        st = dt.datetime(2021, 9, 1, 9, 0, 0)
        ed = dt.datetime(2021, 9, 1, 14, 0, 0)

        self.rad_disp.plot_combined(st, ed, 200, 15)

  
if __name__ == '__main__':
    unittest.main()