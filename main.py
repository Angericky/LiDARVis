import argparse
from code_draw_bev import mapfusion_draw_det_results

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", choices=["det", "trk"], default="det")
    
    args = parser.parse_args()
    return args

def main(args):
    mapfusion_draw_det_results(args.gt)

if __name__ == "__main__":
    args = parse_args()
    main(args)