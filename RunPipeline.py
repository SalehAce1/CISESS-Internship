import argparse
import datetime
import os
import GPMPy as gpy

DT_FORMAT = "%Y/%m/%d-%H:%M:%S"
DT_FORMAT_STR = "%%Y/%%m/%%d-%%H:%%M:%%S"
IMG_PATH = "./combined_images/"
VIDEO_PATH = "./vid_out/video.mp4"
DATA_PATH = "./data/"

def valid_path(s: str):
    if os.path.isdir(s):
        return s
    else:
        msg = "Not a valid path for data: '{0}'.".format(s)
        raise argparse.ArgumentTypeError(msg)

def valid_datetime(s: str):
    try:
        return datetime.datetime.strptime(s, DT_FORMAT)
    except ValueError:
        msg = "Not a valid datetime: '{0}', be sure to use {1}.".format(s, DT_FORMAT_STR)
        raise argparse.ArgumentTypeError(msg)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    
    argparser.add_argument("-u", "--username", help="Username for downloading GPM data", required=True,
                           type=str)
    argparser.add_argument("-p", "--password", help="Password for downloading GPM data", required=True,
                           type=str)
    argparser.add_argument("-st", "--start_dt", help="Start date-time with format: {}".format(DT_FORMAT_STR), required=True,
                           type=valid_datetime)
    argparser.add_argument("-ed", "--end_dt", help="End date-time with format: {}".format(DT_FORMAT_STR), required=True,
                           type=valid_datetime)
    argparser.add_argument("-d", "--data_path", help="Path to the data.", required=False, default=DATA_PATH,
                           type=valid_path)
    argparser.add_argument("-v", "--video_path", help="Path (including the name) of where the video should be placed.", required=False, default=VIDEO_PATH,
                           type=str)
    argparser.add_argument("-f", "--frames", help="The number of frames the animation should be made up of with a default of 100.", required=False, default=100,
                           type=int)
    argparser.add_argument("-fps", "-fps", help="Frames per second of the final video, default of 15.", required=False, default=15,
                           type=int)
    argparser.add_argument("-uc", "--user_cloudsat", help="Username for downloading CloudSat data.", required=False,
                           type=str)
    argparser.add_argument("-pc", "--pass_cloudsat", help="Password for downloading CloudSat data.", required=False,
                           type=str, default=None)
    argparser.add_argument("-force", "--force_combine", help="Forces the code to combine all the plots, including cloudsat if user/pass for it is included.", required=False,
                            type=bool, default=False)
    argparser.add_argument("-db", "--debug", help="Set to true if you want to enable debug logging.", required=False,
                           type=bool, default=False)

    args = argparser.parse_args()

    gpm = gpy.RadarDisplay(args.data_path, args.username, args.password, debug_mode=args.debug)

    if not args.user_cloudsat or not args.pass_cloudsat:
        print("Creating side by side plot animation using only GPM data.")
        gpm.plot_combined(args.start_dt, args.end_dt, args.frames, args.fps)
    elif args.force_combine:
        print("Creating side by side plot using GPM and CloudSat")
        gpm.plot_combined_with_cloudsat(args.start_dt, args.end_dt, args.frames, args.fps)
    else:
        print("Just creating side 2d plot using CloudSat data")
        data = gpm.get_cloudsat_files(args.start_dt, args.end_dt, args.user_cloudsat, args.pass_cloudsat)
        gpm.plot_side_2d(data, args.start_dt, args.end_dt, "2d_images/Output.png")
        