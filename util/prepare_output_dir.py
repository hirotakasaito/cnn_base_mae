import argparse
import datetime
import json
import os
import tempfile

def prepare_output_dir(args, user_specified_dir=None, argv=None,time_format='%Y%m%dT%H%M%S.%f'):
    time_str = datetime.datetime.now().strftime(time_format)

    if user_specified_dir is not None:
        if os.path.exists(user_specified_dir):
            if not os.path.isdir(user_specified_dir):
                raise RuntimeError('{} is not a directory'.format(user_specified_dir))
        outdir = os.path.join(user_specified_dir, time_str)
        if os.path.exists(outdir):
            raise RuntimeError('{} is already exists...'.format(outdir))
        else:
            os.makedirs(outdir)
    else:
        outdir = tempfile.mkdtemp(prefix=time_str)
    return outdir
