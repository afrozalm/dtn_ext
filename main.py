import tensorflow as tf
from model import DTN
from solver import Solver

flags = tf.app.flags
flags.DEFINE_string('mode', 'train', "'pretrain', 'train' or 'eval'")
flags.DEFINE_float('margin', 512.0, "margin for negative class")
flags.DEFINE_float('pos_weight', 1.0, "weight to ucn pos loss")
flags.DEFINE_float('ucn_weight', 5.0, "weight to ucn loss")
flags.DEFINE_float('f_weight', 3.0, "weight to dtn's f-constancy")
flags.DEFINE_integer('n_classes', 200, "number of classes")
flags.DEFINE_integer('batch_size', 25, "set the value of batch_size")
flags.DEFINE_float('reconst_weight', 15.0, "weight for reconstruction loss")
flags.DEFINE_float('learning_rate', 1e-4, "learning rate for Adam")
flags.DEFINE_integer('pretrain_iter', 20000, "iterations to pretrain model")
flags.DEFINE_integer('train_iter', 20000, "iterations to train model")
flags.DEFINE_integer('sample_iter', 100, "iterations to get images")
flags.DEFINE_string('pretrained_model', '',
                    "location of pretrained model")
flags.DEFINE_string('test_model', 'model/dtn-400', "location for test model")
flags.DEFINE_string('model_save_path', 'model',
                    "directory for saving the model")
flags.DEFINE_string('sample_save_path', 'sample',
                    "directory for saving the sampled images")
flags.DEFINE_integer('src_disc_rep', 3, "disc repeats for source part")
flags.DEFINE_integer('src_gen_rep', 1, "gen repeats for source part")
flags.DEFINE_integer('trg_disc_rep', 3, "disc repeats for target part")
flags.DEFINE_integer('trg_gen_rep', 1, "disc repeats for target part")
FLAGS = flags.FLAGS
print FLAGS


def main(_):

    model = DTN(mode=FLAGS.mode,
                learning_rate=FLAGS.learning_rate,
                n_classes=FLAGS.n_classes,
                margin=FLAGS.margin,
                ucn_weight=FLAGS.ucn_weight,
                pos_weight=FLAGS.pos_weight,
                f_weight=FLAGS.f_weight,
                reconst_weight=FLAGS.reconst_weight)

    solver = Solver(model, batch_size=FLAGS.batch_size,
                    pretrain_iter=FLAGS.pretrain_iter,
                    train_iter=FLAGS.train_iter, sample_iter=100,
                    real_dir='real-face', caric_dir='caricature-face',
                    combined_dir='class-combined.pkl',
                    model_save_path=FLAGS.model_save_path,
                    sample_save_path=FLAGS.sample_save_path,
                    pretrained_model=FLAGS.pretrained_model,
                    test_model=FLAGS.test_model,
                    src_disc_rep=FLAGS.src_disc_rep,
                    src_gen_rep=FLAGS.src_gen_rep,
                    trg_disc_rep=FLAGS.trg_disc_rep,
                    trg_gen_rep=FLAGS.trg_gen_rep)

    # create directories if not exist
    if not tf.gfile.Exists(FLAGS.model_save_path):
        tf.gfile.MakeDirs(FLAGS.model_save_path)
    if not tf.gfile.Exists(FLAGS.sample_save_path):
        tf.gfile.MakeDirs(FLAGS.sample_save_path)

    if FLAGS.mode == 'pretrain':
        solver.pretrain()
    elif FLAGS.mode == 'train':
        solver.train()
    else:
        solver.eval()


if __name__ == '__main__':
    tf.app.run()
