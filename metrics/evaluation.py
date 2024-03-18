import tensorflow as tf
import numpy as np

def hypersphere_percentage(y_pred, y_true, class_anchors, radius):
    """
    Compute the percentage of points that are in the hypersphere of the 
    associated anchor.
    """    
    true_anchor = tf.gather(class_anchors, y_true)
    
    # Distance of each point to its anchor
    dist = tf.norm(y_pred - true_anchor, axis=1)
    
    percentage = tf.reduce_mean(tf.cast(tf.less_equal(dist, radius), tf.float32))
    
    return percentage


 