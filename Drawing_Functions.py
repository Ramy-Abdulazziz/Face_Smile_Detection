import numpy as np


def createSquareCornerFrame(x, y, w, h):

    q_width, q_height = 0.25 * w, 0.25 * h
    tq_width, tq_height = 0.75 * w, 0.75 * h

    pts_lcorn_top = np.array([[x + q_width, y],
                              [x, y],
                              [x, y + q_height]],
                             np.int32)

    pts_lcorn_bot = np.array([[x, y + tq_height],
                              [x, y + h],
                              [x + q_width, y + h]],
                             np.int32)

    pts_rcorn_top = np.array([[x + tq_width, y],
                              [x + w, y],
                              [x + w, y + q_height]],
                             np.int32)

    pts_rcorn_bot = np.array([[x + tq_width, y + h],
                              [x + w, y + h],
                              [x + w, y + tq_height]],
                             np.int32)

    frame = [pts_lcorn_top, pts_lcorn_bot,
             pts_rcorn_top, pts_rcorn_bot]

    frame = [x.reshape((-1, 1, 2)) for x in frame]

    return frame
