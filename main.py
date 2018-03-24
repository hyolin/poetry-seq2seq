# -*- coding:utf-8 -*-

from plan import Planner
from predict import Seq2SeqPredictor
import sys

import tensorflow as tf
tf.app.flags.DEFINE_boolean('cangtou', False, 'Generate Acrostic Poem')


def get_cangtou_keywords(ipt):
    assert(len(ipt) == 4)
    return [c for c in ipt]

def main(cangtou=False):
    planner = Planner()
    with Seq2SeqPredictor() as predictor:
        # Run loop
        terminate = False
        while not terminate:
            try:
                ipt = input('ipt Text:\n').strip()

                if not ipt:
                    print('ipt cannot be empty!')
                elif ipt.lower() in ['quit', 'exit']:
                    terminate = True
                else:
                    if cangtou:
                        keywords = get_cangtou_keywords(ipt)
                    else:
                        # Generate keywords
                        keywords = planner.plan(ipt)

                    # Generate poem
                    lines = predictor.predict(keywords)

                    # print(keywords and poem
                    print('Keyword:\t\tPoem:')
                    for line_number in range(4):
                        punctuation = u'，' if line_number % 2 == 0 else u'。'
                        print(u'{keyword}\t\t{line}{punctuation}'.format(
                            keyword=keywords[line_number],
                            line=lines[line_number],
                            punctuation=punctuation
                        ))

            except EOFError:
                terminate = True
            except KeyboardInterrupt:
                terminate = True
    print('\nTerminated.')


if __name__ == '__main__':
    main(cangtou=tf.app.flags.FLAGS.cangtou)
