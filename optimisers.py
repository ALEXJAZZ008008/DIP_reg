# Copyright University College London 2021
# Copyright Harvard Medical School 2021
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# Author: Ludovica Brusaferri, A. Anthinoa Center For Biomedical Imaging, HMS
# For internal research only.


import numpy as np
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
import tensorflow as tf
import tensorflow.keras as k


# https://towardsdatascience.com/custom-optimizer-in-tensorflow-d5b41f75644a
class PowerSign(optimizer.Optimizer):
    """Implementation of PowerSign.
    See [Bello et. al., 2017](https://arxiv.org/abs/1709.07417)
    @@__init__
    """

    def __init__(self, learning_rate=0.001, alpha=0.01, beta=0.5, use_locking=False, name="PowerSign"):
        super(PowerSign, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._alpha = alpha
        self._beta = beta #self._clipnorm = clipnorm + define etc

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._alpha_t = None
        self._beta_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._alpha_t = ops.convert_to_tensor(self._beta, name="alpha_t")
        self._beta_t = ops.convert_to_tensor(self._beta, name="beta_t")

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)

    def _resource_apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        alpha_t = math_ops.cast(self._alpha_t, var.dtype.base_dtype)
        beta_t = math_ops.cast(self._beta_t, var.dtype.base_dtype)

        eps = 1e-7  # cap for moving average

        m = self.get_slot(var, "m")
        m_t = m.assign(tf.maximum(beta_t * m + eps, tf.abs(grad)))

        var_update = state_ops.assign_sub(var, lr_t * grad * tf.exp(
            tf.math.log(alpha_t) * tf.sign(grad) * tf.sign(m_t)))  # Update 'ref' by subtracting 'value
        # Create an op that groups multiple operations.
        # When this op finishes, all ops in input have finished
        return control_flow_ops.group(*[var_update, m_t])

    def _resource_apply_sparse(self, grad, handle, indices):
        raise NotImplementedError("Sparse gradient updates are not supported.")


# https://towardsdatascience.com/custom-optimizer-in-tensorflow-d5b41f75644a
class AddSign(optimizer.Optimizer):
    """Implementation of AddSign.
    See [Bello et. al., 2017](https://arxiv.org/abs/1709.07417)
    @@__init__
    """

    def __init__(self, learning_rate=1.001, alpha=0.01, beta=0.5, use_locking=False, name="AddSign"):
        super(AddSign, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._alpha = alpha
        self._beta = beta

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._alpha_t = None
        self._beta_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._alpha_t = ops.convert_to_tensor(self._beta, name="beta_t")
        self._beta_t = ops.convert_to_tensor(self._beta, name="beta_t")

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)

    def _resource_apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta_t = math_ops.cast(self._beta_t, var.dtype.base_dtype)
        alpha_t = math_ops.cast(self._alpha_t, var.dtype.base_dtype)

        eps = 1e-7  # cap for moving average

        m = self.get_slot(var, "m")
        m_t = m.assign(tf.maximum(beta_t * m + eps, tf.abs(grad)))

        var_update = state_ops.assign_sub(var, lr_t * grad * (1.0 + alpha_t * tf.sign(grad) * tf.sign(m_t)))
        # Create an op that groups multiple operations
        # When this op finishes, all ops in input have finished
        return control_flow_ops.group(*[var_update, m_t])

    def _resource_apply_sparse(self, grad, handle, indices):
        raise NotImplementedError("Sparse gradient updates are not supported.")


# from https://github.com/Smokrow/addons/blob/dev/adafactore/tensorflow_addons/optimizers/adafactor.py
def reduce_rms(x):
    return tf.math.sqrt(tf.reduce_mean(tf.square(x)))


class AdafactorOptimizer(tf.keras.optimizers.Optimizer):
    """Optimizer that implements the Adafactor algorithm.
    Adafactor is described in https://arxiv.org/abs/1804.04235.
    Adafactor is most similar to Adam (Kingma and Ba), the major differences are:
    1. For a two-dimensional AxB weight matrix, Adafactor uses only A+B auxiliary
         parameters to maintain the second-moment estimator, instead of AB.
         This is advantageous on memory-limited systems.    In addition, beta1
         (momentum) is set to zero by default, saving an additional auxiliary
         parameter per weight.    Variables with >=3 dimensions are treated as
         collections of two-dimensional matrices - factorization is over the final
         two dimensions.
    2. Adafactor incorporates "update-clipping" - a scale-invariant analog of
         gradient clipping.  This adds stability
    3. Adafactor does not require an external "learning rate".    By default, it
         incorporates a relative-update-scale schedule, corresponding to
         inverse-square-root learning-rate-decay in ADAM.  We hope this works well
         for most applications.
    ALGORITHM:
    parameter -= absolute_update_scale * clip(grad / grad_scale)
    where:
        absolute_update_scale := relative_update_scale * parameter_scale
        relative_update_scale := min((step_num + 1)**-0.5, 1e-2)
        parameter_scale := max(rms(var)), epsilon2)
        clip(x) := x / max(1.0, rms(x))
        grad_scale := tf.sqrt(v)     (v is the second-moment estimator)
    The second-moment estimator v is maintained in a manner similar to Adam:
    We initialize
    ```
    if var is 2-dimensional:
        v_r <- zeros([num_rows])
        v_c <- zeros([num_cols])
    if var is 0-dimensional or 1-dimensional:
        v <- zeros(shape(var))
    ```
    The update rule is as follows:
    ```
    decay_rate = 1 - (step_num + 1) ^ -0.8
    grad_squared = tf.square(grad) + epsilon1
    if var is 2-dimensional:
        v_r <- decay_rate * v_r + (1 - decay_rate) * \
                                   reduce_mean(grad_squared, 1)
        v_c <- decay_rate * v_c + (1 - decay_rate) * \
                                   reduce_mean(grad_squared, 0)
        v = outer_prod(v_r, v_c) / reduce_mean(v_r)
    if var is 0-dimensional or 1-dimensional:
        v <- decay_rate * v + (1 - decay_rate) * grad_squared
    ```
    For variables with >=3 dimensions, we factorize the second-moment accumulator
    over the final 2 dimensions.    See the code for details.
    Several parts of this algorithm are configurable from the initializer.
        multiply_by_parameter_scale:    If True, then compute absolute_update_scale
            as described above.  If False, let absolute_update_scale be the externally
            supplied learning_rate.
        learning_rate: represents relative_update_scale if
            multiply_by_parameter_scale==True, or absolute_update_scale if
            multiply_by_parameter_scale==False.
        decay_rate: Decay rate of the second moment estimator (varies by step_num).
            This should be set to a function such that:
            1-1/(step_num + 1) <= decay_rate(step_num) < 1.0
        beta1: enables momentum, as in Adam.    Uses extra memory if nonzero.
        clipping_threshold: should be >=1.0 or None for no update clipping
        factored: whether to factor the second-moment estimator.    True means
            less memory usage.
    """

    def __init__(self,
                 multiply_by_parameter_scale=False,
                 learning_rate=None,
                 decay_rate=None,
                 beta1=0.0,
                 clipping_threshold=1.0,
                 factored=True,
                 use_locking=False,
                 name="Adafactor",
                 epsilon1=1e-30,
                 epsilon2=1e-3,
                 **kwargs):
        """Construct a new Adafactor optimizer.
        See class comment.
        Args:
            multiply_by_parameter_scale: a boolean
            learning_rate: an optional Scalar.
            decay_rate: an optional Scalar.
            beta1: a float value between 0 and 1
            clipping_threshold: an optional float >= 1
            factored: a boolean - whether to use factored second-moment estimator
                for 2d variables
            use_locking: If True use locks for update operations.
            name: Optional name for the operations created when applying gradients.
                Defaults to "AdafactorOptimizer".
            epsilon1: Regularization constant for squared gradient.
            epsilon2: Regularization constant for parameter scale.
        Raises:
            ValueError: if absolute_update_scale and relative_update_scale_fn are both
                present or both absent.
        """
        super(AdafactorOptimizer, self).__init__(name=name, **kwargs)

        # Set Flags
        self.multiply_by_parameter_scale = multiply_by_parameter_scale
        self.factored = factored
        self.use_locking = use_locking
        self.has_beta_1 = (beta1 != 0.0)

        # Set defaults
        if learning_rate is None:
            learning_rate = self._learning_rate_default(
                multiply_by_parameter_scale)

        if decay_rate is None:
            decay_rate = self._decay_rate_default()

        # Set Hypers
        self._set_hyper("decay_rate", decay_rate)
        self._set_hyper("learning_rate", learning_rate)
        self._set_hyper("beta1", beta1)
        self._set_hyper("clipping_threshold", clipping_threshold)
        self._set_hyper("factored", factored)
        self._set_hyper("epsilon1", epsilon1)
        self._set_hyper("epsilon2", epsilon2)

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super()._prepare_local(var_device, var_dtype, apply_state)

        local_step = math_ops.cast(self.iterations + 1, var_dtype)
        learning_rate_t = tf.identity(
            self._get_hyper("learning_rate", var_dtype))
        decay_rate_t = tf.identity(self._get_hyper("decay_rate", var_dtype))
        beta_1_t = tf.identity(self._get_hyper("beta1", var_dtype))
        clipping_threshold_t = tf.identity(
            self._get_hyper("clipping_threshold", var_dtype))
        epsilon1_t = tf.identity(self._get_hyper("epsilon1", var_dtype))
        epsilon2_t = tf.identity(self._get_hyper("epsilon2", var_dtype))

        apply_state[(var_device, var_dtype)].update(
            dict(
                learning_rate=learning_rate_t,
                decay_rate=decay_rate_t,
                beta1=beta_1_t,
                clipping_threshold=clipping_threshold_t,
                epsilon1=epsilon1_t,
                epsilon2=epsilon2_t,
            )
        )

    def get_config(self):
        config = {
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "decay_rate": self._serialize_hyperparameter("decay_rate"),
            "beta1": self._serialize_hyperparameter("beta1"),
            "clipping_threshold": self._serialize_hyperparameter("clipping_threshold"),
            "epsilon1": self._serialize_hyperparameter("epsilon1"),
            "epsilon2": self._serialize_hyperparameter("epsilon2")
        }
        base_config = super(AdafactorOptimizer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _should_use_factored_second_moment_estimate(self, shape):
        """Should we use a factored second moment estimator.
        Based on the shape of the variable.
        Args:
            shape: a list of integers
        Returns:
            a boolean
        """
        return self.factored and len(shape) >= 2

    def _create_slots(self, var_list):
        for var in var_list:
            shape = var.get_shape().as_list()
            if self.has_beta_1:
                self.add_slot(var, "m")
            if self._should_use_factored_second_moment_estimate(shape):
                r_val = tf.zeros(shape[:-1], dtype=tf.float32)
                c_val = tf.zeros(shape[:-2] + shape[-1:], dtype=tf.float32)
                self.add_slot(var, "vr", initializer=r_val)
                self.add_slot(var, "vc", initializer=c_val)
            else:
                v_val = tf.zeros(shape, dtype=tf.float32)
                self.add_slot(var, "v", initializer=v_val)

    def _apply_dense(self, grad, var):
        return self._resource_apply_dense(grad, var)

    def _apply_sparse(self, grad, var):
        return self._apply_dense(tf.convert_to_tensor(grad), var)

    def _resource_apply_sparse(self, grad, handle, indices, apply_state):
        return self._resource_apply_dense(
            tf.convert_to_tensor(tf.IndexedSlices(
                grad, indices, tf.shape(handle))),
            handle)

    def _parameter_scale(self, var):
        """Estimate the scale of the parameters from the current values.
        We include a minimum value of 0.001 to give it a chance to escape 0
        if it was zero-initialized.
        Instead of using the value, we could impute the scale from the shape,
        as initializers do.
        Args:
            var: a variable or Tensor.
        Returns:
            a Scalar
        """
        tf.cast(var, dtype=tf.float32)
        testy = tf.maximum(reduce_rms(var), self._get_hyper("epsilon2"))
        tf.cast(testy, dtype=tf.float32)
        return tf.maximum(reduce_rms(var), self._get_hyper("epsilon2"))

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)

        grad = tf.cast(grad, dtype=tf.float32)
        grad_squared = tf.square(grad) + coefficients["epsilon1"]
        grad_squared_mean = tf.reduce_mean(grad_squared)
        decay_rate = coefficients["decay_rate"]
        update_scale = coefficients["learning_rate"]
        old_val = var
        if self.multiply_by_parameter_scale:
            scale_factor = self._parameter_scale(old_val)
            update_scale *= tf.cast(scale_factor, dtype=tf.float32)
        # HACK: Make things dependent on grad.
        # This confounds the XLA rewriter and keeps it from fusing computations
        # across different variables.  This fusion is a bad for HBM usage, since
        # it causes the gradients to persist in memory.
        decay_rate += grad_squared_mean * 1e-30
        update_scale += grad_squared_mean * 1e-30
        # END HACK
        mixing_rate = 1.0 - decay_rate
        shape = var.get_shape().as_list()
        updates = []
        if self._should_use_factored_second_moment_estimate(shape):
            grad_squared_row_mean = tf.reduce_mean(grad_squared, -1)
            vr = self.get_slot(var, "vr")
            new_vr = (decay_rate * vr + mixing_rate * grad_squared_row_mean)
            vr_update = vr.assign(new_vr, use_locking=self.use_locking)
            updates.append(vr_update)

            grad_squared_col_mean = tf.reduce_mean(grad_squared, -2)
            vc = self.get_slot(var, "vc")
            new_vc = (decay_rate * vc + mixing_rate * grad_squared_col_mean)
            vc_update = vc.assign(new_vc, use_locking=self.use_locking)
            updates.append(vc_update)

            long_term_mean = tf.reduce_mean(new_vr, -1, keepdims=True)
            r_factor = tf.math.rsqrt(new_vr / long_term_mean)
            c_factor = tf.math.rsqrt(new_vc)
            x = grad * tf.expand_dims(r_factor, -1) * \
                tf.expand_dims(c_factor, -2)
        else:
            v = self.get_slot(var, "v")
            new_v = decay_rate * v + mixing_rate * grad_squared
            v_update = v.assign(new_v, use_locking=self.use_locking)
            updates = [v_update]
            x = grad * tf.math.rsqrt(new_v)

        if coefficients["clipping_threshold"] is not None:
            clipping_denom = tf.maximum(1.0, reduce_rms(
                x) / coefficients["clipping_threshold"])
            x /= clipping_denom
        subtrahend = update_scale * x

        if self.has_beta_1:
            m = self.get_slot(var, "m")
            new_m = coefficients["beta1"] * \
                tf.cast(m, dtype=tf.float32) + \
                (1.0 - coefficients["beta1"]) * subtrahend
            subtrahend = new_m
            new_m = self._cast_like(new_m, var)
            m_update_value = m.assign(new_m, use_locking=self.use_locking)
            updates.append(m_update_value)

        new_val = tf.cast(old_val, dtype=tf.float32) - subtrahend
        new_val = var.assign(new_val, use_locking=self.use_locking)
        updates = [new_val] + updates
        return tf.group(*updates)

    def _cast_like(self, x, y):
        """Cast x to y's dtype, if necessary. Grabbed from tensor2tensor/layers/common_layers"""
        x = tf.convert_to_tensor(x)
        y = tf.convert_to_tensor(y)

        if x.dtype.base_dtype == y.dtype.base_dtype:
            return x

        cast_x = tf.cast(x, dtype=y.dtype)
        if cast_x.device != x.device:
            x_name = "(eager Tensor)"
            try:
                x_name = x.name
            except AttributeError:
                pass
            # tf.logging.warning("Cast for %s may induce copy from '%s' to '%s'", x_name,
            #                    x.device, cast_x.device)
            return cast_x

    def _decay_rate_default(self):
        return self._adafactor_decay_rate_pow(0.8)

    def _learning_rate_default(self, multiply_by_parameter_scale):
        learning_rate = tf.minimum(tf.math.rsqrt(self.step_num() + 1.0), 0.01)
        if not multiply_by_parameter_scale:
            learning_rate *= 0.05
        return learning_rate

    def _adafactor_decay_rate_adam(self, beta2):
        """Second-moment decay rate like Adam, subsuming the correction factor.
        Args:
            beta2: a float between 0 and 1
        Returns:
            a scalar
        """
        t = tf.cast(self.iterations, dtype=tf.float32) + 1.0
        decay = beta2 * (1.0 - tf.pow(beta2, t - 1.0)) / \
            (1.0 - tf.pow(beta2, t))
        # decay = tf.cond(tf.equal(t, 1.0), lambda: beta2, lambda: decay)
        return decay

    def _adafactor_decay_rate_pow(self, exponent):
        """Second moment decay rate where memory-length grows as step_num^exponent.
        Args:
            exponent: a float between 0 and 1
        Returns:
            a scalar
        """
        return 1.0 - tf.pow((self.step_num() + 1.0), -exponent)

    def step_num(self):
        return tf.cast(self.iterations, dtype=tf.float32)

# from: https://github.com/huggingface/transformers/blob/master/src/transformers/optimization_tf.py


class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Applys a warmup schedule on a given learning rate decay schedule."""

    def __init__(self, initial_learning_rate, decay_schedule_fn, warmup_steps, power=1.0, name=None):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.power = power
        self.decay_schedule_fn = decay_schedule_fn
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "WarmUp") as name:
            # Implements polynomial warmup. i.e., if global_step < warmup_steps, the
            # learning rate will be `global_step/num_warmup_steps * init_lr`.
            global_step_float = tf.cast(step, dtype=tf.float32)
            warmup_steps_float = tf.cast(self.warmup_steps, dtype=tf.float32)
            warmup_percent_done = global_step_float / warmup_steps_float
            warmup_learning_rate = self.initial_learning_rate * \
                tf.math.pow(warmup_percent_done, self.power)
            return tf.cond(
                global_step_float < warmup_steps_float,
                lambda: warmup_learning_rate,
                lambda: self.decay_schedule_fn(step),
                name=name,
            )

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_schedule_fn": self.decay_schedule_fn,
            "warmup_steps": self.warmup_steps,
            "power": self.power,
            "name": self.name,
        }


class WarmUpLinearDecayScheduler(k.callbacks.Callback):
    """Cosine decay with warmup learning rate scheduler
    """

    def __init__(self,
                 learning_rate_base,
                 total_steps,
                 global_step_init=0,
                 warmup_learning_rate=0.0,
                 warmup_steps=0,
                 hold_base_rate_steps=0,
                 verbose=0):
        """Constructor for cosine decay with warmup learning rate scheduler.
    Arguments:
        learning_rate_base {float} -- base learning rate.
        total_steps {int} -- total number of training steps.
    Keyword Arguments:
        global_step_init {int} -- initial global step, e.g. from previous checkpoint.
        warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})
        warmup_steps {int} -- number of warmup steps. (default: {0})
        hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                    before decaying. (default: {0})
        verbose {int} -- 0: quiet, 1: update messages. (default: {0})
        """

        super(WarmUpLinearDecayScheduler, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.global_step = global_step_init
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.hold_base_rate_steps = hold_base_rate_steps
        self.verbose = verbose
        self.learning_rates = []

        learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=learning_rate_base, decay_steps=total_steps, end_learning_rate=0.0
        )

        self.sched = WarmUp(learning_rate_base,
                            learning_rate_fn, warmup_steps=warmup_steps)

    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        lr = k.backend.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    def on_batch_begin(self, batch, logs=None):

        # lr = cosine_decay_with_warmup(global_step=self.global_step,
        #                               learning_rate_base=self.learning_rate_base,
        #                               total_steps=self.total_steps,
        #                               warmup_learning_rate=self.warmup_learning_rate,
        #                               warmup_steps=self.warmup_steps,
        #                               hold_base_rate_steps=self.hold_base_rate_steps)

        lr = self.sched(self.global_step)

        k.backend.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nBatch %05d: setting learning '
                  'rate to %s.' % (self.global_step + 1, lr))


# https://github.com/nnormandin/YellowFin_Keras/blob/master/yellowfin.py
# Values for gate_gradients.
GATE_NONE = 0
GATE_OP = 1
GATE_GRAPH = 2


class YFOptimizer(object):
    def __init__(self, learning_rate=0.1, momentum=0.0, clip_thresh=None, beta=0.999, curv_win_width=20,
                 mu_update_interval=1, zero_debias=True, delta_mu=0.0):
        '''
        clip thresh is the threshold value on ||lr * gradient||
        delta_mu can be place holder/variable/python scalar. They are used for additional
        momentum in situations such as asynchronous-parallel training. The default is 0.0
        for basic usage of the optimizer.
        Args:
          learning_rate: python scalar. The initial value of learning rate, we use 1.0 in our paper.
          momentum: python scalar. The initial value of momentum, we use 0.0 in our paper.
          clip_thresh: python scalar. The cliping threshold for tf.clip_by_global_norm.
            if None, no clipping will be carried out.
          beta: python scalar. The smoothing parameter for estimations.
          delta_mu: for extensions. Not necessary in the basic use.
        Other features:
          If you want to manually control the learning rates, self.lr_factor is
          an interface to the outside, it is an multiplier for the internal learning rate
          in YellowFin. It is helpful when you want to do additional hand tuning
          or some decaying scheme to the tuned learning rate in YellowFin.
          Example on using lr_factor can be found here:
          https://github.com/JianGoForIt/YellowFin/blob/master/char-rnn-tensorflow/train_YF.py#L140
        '''
        self._lr = learning_rate
        self._mu = momentum

        self._lr_var = tf.Variable(learning_rate, dtype=tf.float32, name="YF_lr", trainable=False)
        self._mu_var = tf.Variable(momentum, dtype=tf.float32, name="YF_mu", trainable=False)
        # for step scheme or decaying scheme for the learning rates
        self.lr_factor = tf.Variable(1.0, dtype=tf.float32, name="YF_lr_factor", trainable=False)
        if clip_thresh is not None:
            self._clip_thresh_var = tf.Variable(clip_thresh, dtype=tf.float32, name="YF_clip_thresh", trainable=False)
        else:
            self._clip_thresh_var = None

        # the underlying momentum optimizer
        self._optimizer = tf.compat.v1.train.MomentumOptimizer(self._lr_var * self.lr_factor,
                                                               tf.cast(self._mu_var, dtype=tf.float32) + delta_mu)

        # moving average for statistics
        self._beta = beta
        self._moving_averager = None

        # for global step counting
        self._global_step = tf.Variable(0, trainable=False)

        # for conditional tuning
        self._do_tune = tf.greater(self._global_step, tf.constant(0))

        self._zero_debias = zero_debias

        self._tvars = None

        # for curvature range
        self._curv_win_width = curv_win_width
        self._curv_win = None

    def curvature_range(self):
        # set up the curvature window
        self._curv_win = \
            tf.Variable(np.zeros([self._curv_win_width, ]), dtype=tf.float32, name="curv_win", trainable=False)
        self._curv_win = tf.compat.v1.scatter_update(self._curv_win,
                                                     tf.cast(self._global_step, dtype=tf.int32) % self._curv_win_width,
                                                     self._grad_norm_squared)
        # note here the iterations start from iteration 0
        valid_window = tf.slice(self._curv_win, tf.constant([0, ]),
                                tf.expand_dims(tf.minimum(tf.constant(self._curv_win_width),
                                                          tf.cast(self._global_step, dtype=tf.int32) + 1), dim=0))
        self._h_min_t = tf.reduce_min(valid_window)
        self._h_max_t = tf.reduce_max(valid_window)

        curv_range_ops = []
        with tf.control_dependencies([self._h_min_t, self._h_max_t]):
            avg_op = self._moving_averager.apply([self._h_min_t, self._h_max_t])
            with tf.control_dependencies([avg_op]):
                self._h_min = tf.identity(self._moving_averager.average(self._h_min_t))
                self._h_max = tf.identity(self._moving_averager.average(self._h_max_t))
        curv_range_ops.append(avg_op)
        return curv_range_ops

    def grad_variance(self):
        grad_var_ops = []
        tensor_to_avg = []
        for t, g in zip(self._tvars, self._grads):
            if isinstance(g, ops.IndexedSlices):
                tensor_to_avg.append(
                    tf.reshape(tf.compat.v1.unsorted_segment_sum(g.values, g.indices, g.dense_shape[0]),
                               shape=t.get_shape()))
            else:
                tensor_to_avg.append(g)
        avg_op = self._moving_averager.apply(tensor_to_avg)
        grad_var_ops.append(avg_op)
        with tf.control_dependencies([avg_op]):
            self._grad_avg = [self._moving_averager.average(val) for val in tensor_to_avg]
            self._grad_avg_squared = [tf.square(val) for val in self._grad_avg]
        self._grad_var = self._grad_norm_squared_avg - tf.add_n([tf.reduce_sum(val) for val in self._grad_avg_squared])
        return grad_var_ops

    def dist_to_opt(self):
        dist_to_opt_ops = []
        # running average of the norm of gradeint
        self._grad_norm = tf.sqrt(self._grad_norm_squared)
        avg_op = self._moving_averager.apply([self._grad_norm, ])
        dist_to_opt_ops.append(avg_op)
        with tf.control_dependencies([avg_op]):
            self._grad_norm_avg = self._moving_averager.average(self._grad_norm)
            # single iteration distance estimation, note here self._grad_norm_avg is per variable
            self._dist_to_opt = self._grad_norm_avg / self._grad_norm_squared_avg
        # running average of distance
        avg_op = self._moving_averager.apply([self._dist_to_opt])
        dist_to_opt_ops.append(avg_op)
        with tf.control_dependencies([avg_op]):
            self._dist_to_opt_avg = tf.identity(self._moving_averager.average(self._dist_to_opt))
        return dist_to_opt_ops

    def after_apply(self):
        self._moving_averager = tf.train.ExponentialMovingAverage(decay=self._beta, zero_debias=self._zero_debias)
        assert self._grads != None and len(self._grads) > 0
        after_apply_ops = []

        # get per var g**2 and norm**2
        self._grad_squared = []
        self._grad_norm_squared = []
        for v, g in zip(self._tvars, self._grads):
            with ops.colocate_with(v):
                self._grad_squared.append(tf.square(g))
        self._grad_norm_squared = [tf.reduce_sum(grad_squared) for grad_squared in self._grad_squared]

        # the following running average on squared norm of gradient is shared by grad_var and dist_to_opt
        avg_op = self._moving_averager.apply(self._grad_norm_squared)
        with tf.control_dependencies([avg_op]):
            self._grad_norm_squared_avg = [self._moving_averager.average(val) for val in self._grad_norm_squared]
            self._grad_norm_squared = tf.add_n(self._grad_norm_squared)
            self._grad_norm_squared_avg = tf.add_n(self._grad_norm_squared_avg)
        after_apply_ops.append(avg_op)

        with tf.control_dependencies([avg_op]):
            curv_range_ops = self.curvature_range()
            after_apply_ops += curv_range_ops
            grad_var_ops = self.grad_variance()
            after_apply_ops += grad_var_ops
            dist_to_opt_ops = self.dist_to_opt()
            after_apply_ops += dist_to_opt_ops

        return tf.group(*after_apply_ops)

    def get_lr_tensor(self):
        lr = (1.0 - tf.sqrt(self._mu)) ** 2 / self._h_min
        return lr

    def get_mu_tensor(self):
        const_fact = self._dist_to_opt_avg ** 2 * self._h_min ** 2 / 2 / self._grad_var
        coef = tf.Variable([-1.0, 3.0, 0.0, 1.0], dtype=tf.float32, name="cubic_solver_coef")
        coef = tf.compat.v1.scatter_update(coef, tf.constant(2), -(3 + const_fact))
        roots = tf.compat.v1.py_func(np.roots, [coef], Tout=tf.complex64, stateful=False)

        # filter out the correct root
        root_idx = tf.logical_and(tf.logical_and(tf.greater(tf.compat.v1.real(roots), tf.constant(0.0)),
                                                 tf.less(tf.compat.v1.real(roots), tf.constant(1.0))),
                                  tf.less(tf.abs(tf.compat.v1.imag(roots)), 1e-5))
        # in case there are two duplicated roots satisfying the above condition
        root = tf.reshape(tf.gather(tf.gather(roots, tf.where(root_idx)), tf.constant(0)), shape=[])
        tf.assert_equal(tf.size(root), tf.constant(1))

        dr = self._h_max / self._h_min
        mu = tf.maximum(tf.compat.v1.real(root) ** 2, ((tf.sqrt(dr) - 1) / (tf.sqrt(dr) + 1)) ** 2)
        return mu

    def update_hyper_param(self):
        assign_hyper_ops = []
        self._mu = tf.identity(tf.cond(self._do_tune, lambda: self.get_mu_tensor(),
                                       lambda: self._mu_var))
        with tf.control_dependencies([self._mu]):
            self._lr = tf.identity(tf.cond(self._do_tune, lambda: self.get_lr_tensor(),
                                           lambda: self._lr_var))

        with tf.control_dependencies([self._mu, self._lr]):
            self._mu = self._beta * tf.cast(self._mu_var, dtype=tf.float32) + (1 - self._beta) * self._mu
            self._lr = self._beta * tf.cast(self._lr_var, dtype=tf.float32) + (1 - self._beta) * self._lr
            assign_hyper_ops.append(tf.compat.v1.assign(self._mu_var, self._mu))
            assign_hyper_ops.append(tf.compat.v1.assign(self._lr_var, self._lr))
        assign_hyper_op = tf.group(*assign_hyper_ops)
        return assign_hyper_op

    def apply_gradients(self, grads_tvars, global_step):
        self._grads, self._tvars = zip(*grads_tvars)
        with tf.compat.v1.variable_scope("apply_updates"):
            if self._clip_thresh_var is not None:
                self._grads_clip, self._grads_norm = tf.clip_by_global_norm(self._grads, self._clip_thresh_var)
                apply_grad_op = \
                    self._optimizer.apply_gradients(zip(self._grads_clip, self._tvars))
            else:
                apply_grad_op = \
                    self._optimizer.apply_gradients(zip(self._grads, self._tvars))

        with tf.compat.v1.variable_scope("after_apply"):
            after_apply_op = self.after_apply()

        with tf.compat.v1.variable_scope("update_hyper"):
            with tf.control_dependencies([after_apply_op]):
                update_hyper_op = self.update_hyper_param()

        with tf.control_dependencies([update_hyper_op]):
            self._increment_global_step_op = tf.compat.v1.assign(self._global_step,
                                                                 tf.cast(global_step, dtype=tf.int32))

        return tf.group(apply_grad_op, after_apply_op, update_hyper_op, self._increment_global_step_op)

    def compute_gradients(self, loss, var_list=None, gate_gradients=GATE_OP,
                          aggregation_method=None,
                          colocate_gradients_with_ops=False,
                          grad_loss=None):
        return (self._optimizer.compute_gradients(
            loss=loss, var_list=var_list, gate_gradients=gate_gradients,
            aggregation_method=aggregation_method,
            colocate_gradients_with_ops=colocate_gradients_with_ops,
            grad_loss=grad_loss))

    def minimize(self, loss, global_step=None, var_list=None,
                 gate_gradients=GATE_OP, aggregation_method=None,
                 colocate_gradients_with_ops=False, name=None,
                 grad_loss=None):
        """Adapted from Tensorflow Optimizer base class member function:
        Add operations to minimize `loss` by updating `var_list`.
        This method simply combines calls `compute_gradients()` and
        `apply_gradients()`. If you want to process the gradient before applying
        them call `tf.gradients()` and `self.apply_gradients()` explicitly instead
        of using this function.
        """
        grads_and_vars = self._optimizer.compute_gradients(
            loss, var_list=var_list, gate_gradients=gate_gradients,
            aggregation_method=aggregation_method,
            colocate_gradients_with_ops=colocate_gradients_with_ops,
            grad_loss=grad_loss)

        vars_with_grad = [v for g, v in grads_and_vars if g is not None]
        if not vars_with_grad:
            raise ValueError(
                "No gradients provided for any variable, check your graph for ops"
                " that do not support gradients, between variables %s and loss %s." %
                ([str(v) for _, v in grads_and_vars], loss))
        for g, v in grads_and_vars:
            print("g ", g)
            print("v ", v)

        return self.apply_gradients(grads_and_vars, global_step)
