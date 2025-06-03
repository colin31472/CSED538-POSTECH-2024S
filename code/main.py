from absl import app, flags
from ml_collections import config_flags
from image_main import main as main_img
from save_results import save_results

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=True)

def main(argv):
  if 'image' in FLAGS.config.task:
    result = main_img(FLAGS.config)
  else:
    raise ValueError("Invalid subject; target subject arguments should be either image")
  save_results(result, FLAGS.config)

if __name__ == '__main__':
  flags.mark_flags_as_required(['config'])
  app.run(main)