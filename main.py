import argparse
from code_draw_bev import mapfusion_draw_det_results

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", choices=["det", "trk"], default="det")
    parser.add_argument("--show_ids", type=bool, default=True, help="show the object id in image")
    parser.add_argument("--show_color", action='store_true')
    args = parser.parse_args()
    return args

def main(args):
    mapfusion_draw_det_results(args.gt, args.show_ids, args.show_color)

if __name__ == "__main__":
    args = parse_args()
    main(args)