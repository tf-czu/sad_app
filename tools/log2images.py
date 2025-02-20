import os

from osgar.logger import LogReader, lookup_stream_id
from osgar.lib.serialize import deserialize


def main(log_file, im_stream, out_dir):
    only_im = lookup_stream_id(log_file, im_stream)

    with LogReader(log_file, only_stream_id=only_im) as log:
        ii = 0
        for timestamp, stream_id, data in log:
            if stream_id == only_im:
                jpeg_im = deserialize(data)
                if ii%50 == 0:
                    with open(os.path.join(out_dir, f"im_{ii:05d}.jpg"), "wb") as f:
                        f.write(jpeg_im)
                ii += 1

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logfile', help='path to logfile')
    parser.add_argument('--im', help='image stream', default="oak_camera.color")
    parser.add_argument('--out', help='out directory', default="logs/tmp_im")
    args = parser.parse_args()

    main(args.logfile, args.im, args.out)
