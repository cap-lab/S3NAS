import argparse
import json

from graph.blockargs import BlockArgsDecoder
from graph.stage import _get_conv_cls
from util.io_utils import load_json_as_attrdict
from util.string_utils import get_filename

parser = argparse.ArgumentParser(description="")
# TODO: maybe we can add to test a folder later. done scripts can be moved to other folder like 'done'
parser.add_argument('model_json', type=str, help="model json file")
parser.add_argument('--suffix', type=str, default="", help="suffix of result file")
parser.add_argument('--mode', choices=['stage_i', 'select_by_rank'], default='stage_i', type=str,
                    help="stage_i : specify stage_i and block_i"
                         "select_by_rank : select the specific block from given rank file")

parser.add_argument('--se_ratio', type=float, default=None, help='se_ratio you want to set')
parser.add_argument('--act_fn', type=str, default=None, help='act_fn you want to set')
parser.add_argument('--global_act_fn', type=str, default=None, help='act_fn you want to set globally')

parser.add_argument('--rank_json', type=str, help="json file for rankings of se blocks. use se_check_csv to get it")
parser.add_argument('--set_blocks_num', type=int, help="how many blocks you want to remove")

parser.add_argument('--set_stage_i', type=str, default=None, help="stage_i to set se and act")
parser.add_argument('--set_block_i', type=str, default=None, help="block_i to set se and act")
parser.add_argument('--set_conv_type', type=str, default=None, help="Set this if you want to set only specific conv type"
                                                                  "Only used for stage_i")


def set_and_print(model_args, stage_i, block_i, source_obj, attr):
    block = model_args.stages_args[stage_i].blocks_args[block_i]
    value = getattr(source_obj, attr)
    setattr(block, attr, value)
    print("setting stage_%d_block_%d with %s %s" % (stage_i, block_i, attr, str(value)))


if __name__ == '__main__':
    args = parser.parse_args()

    model_json = args.model_json
    model_args = load_json_as_attrdict(model_json)

    decoder = BlockArgsDecoder()
    model_args.stages_args = decoder.decode_to_stages_args(model_args.stages_args)
    for stage_args in model_args.stages_args:
        stage_args.blocks_args = decoder.span_blocks_args(stage_args.blocks_args)

    if args.set_conv_type:
        assert args.mode == 'stage_i'

    if args.mode == 'stage_i':
        assert args.set_stage_i is not None

        stage_is = [int(i) for i in args.set_stage_i.split(',')]
        if args.set_block_i is not None:
            block_is = [int(i) for i in args.set_block_i.split(',')]

        for stage_i, stage_args in enumerate(model_args.stages_args):
            if stage_i in stage_is:
                for block_i, block_args in enumerate(stage_args.blocks_args):
                    if args.set_conv_type is not None:
                        if args.set_conv_type != _get_conv_cls(block_args.conv_type).__name__:
                            continue
                    if args.set_block_i is not None:
                        if block_i not in block_is:
                            continue

                    if args.se_ratio is not None:
                        set_and_print(model_args, stage_i, block_i, args, 'se_ratio')
                    if args.act_fn is not None:
                        set_and_print(model_args, stage_i, block_i, args, 'act_fn')

    elif args.mode == 'select_by_rank':
        ranks = load_json_as_attrdict(args.rank_json)
        for i, rank_stagei_blocki in enumerate(ranks):
            if i >= args.set_blocks_num:
                break
            stage_i, block_i = rank_stagei_blocki.stage_i, rank_stagei_blocki.block_i

            if args.se_ratio is not None:
                set_and_print(model_args, stage_i, block_i, args, 'se_ratio')
            if args.act_fn is not None:
                set_and_print(model_args, stage_i, block_i, args, 'act_fn')

    else:
        raise NotImplementedError

    if args.global_act_fn:
        print("setting global act_fn as %s " % args.global_act_fn)
        model_args.act_fn = args.global_act_fn

    suffix = args.suffix
    save_filename = get_filename(model_json) + suffix + '.json'
    print("saving to ", save_filename)
    json.dump(model_args, open(save_filename, "w"), indent=4)
