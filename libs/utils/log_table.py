"""The training log: one table row per epoch."""


# One log row per epoch, columns in groups. Each group: (title, columns); each column:
# (header, metric key, width, format). The use_*_columns functions below add optional columns.
GROUPS = [
    ('train loss', [('event', 'ev_loss', 6, '.3f'), ('reg', 'reg_loss', 6, '.3f'), ('iou', 'iou_loss', 6, '.3f'),
                    ('embed', 'emb_loss', 6, '.3f'), ('span', 'span_loss', 6, '.3f'),
                    ('total', 'final_loss', 6, '.3f')]),
    ('segmentation (val)', [('R@0.3', 'R@0.3', 6, '.2f'), ('R@0.5', 'R@0.5', 6, '.2f'),
                            ('R@0.7', 'R@0.7', 6, '.2f'), ('mIoU', 'mIoU', 6, '.2f'),
                            ('seg/vid', 'seg/vid', 7, '.2f')]),
    ('captioning (val)', [('cos_gt', 'cos_gt', 6, '.3f'), ('cos_rand', 'cos_rand', 8, '.3f'),
                          ('txt_sim', 'txt_sim', 7, '.3f'), ('CIDEr', 'CIDEr', 6, '.2f')]),
]
LEAD = '{:>6} {:>5} {:>8}'   # epoch, seconds, learning rate
HEAD = RULE = ''             # header line and separator, rebuilt whenever GROUPS changes


def _cells(fn):
    """fn(column) for every column, columns joined by spaces and groups by ' | '."""
    return ' | '.join(' '.join(fn(c) for c in cols) for _, cols in GROUPS)


def _refresh_header():
    global HEAD, RULE
    HEAD = LEAD.format('epoch', 'time', 'lr') + ' | ' + _cells(lambda c: '%*s' % (c[2], c[0]))
    RULE = '-' * (len(HEAD) + 4)


_refresh_header()


def _add_loss_columns(*cols):
    """Insert train-loss columns just before the 'total' column."""
    GROUPS[0][1][-1:-1] = list(cols)
    _refresh_header()


def use_ground_columns():
    """Add the grounding loss and the grounding scores to the table."""
    _add_loss_columns(('ground', 'ground_loss', 6, '.3f'))
    GROUPS.append(('grounding (val)', [('R1@.5', 'G R1@0.5', 6, '.2f'), ('R1@.7', 'G R1@0.7', 6, '.2f'),
                                       ('mIoU', 'G mIoU', 6, '.2f')]))
    _refresh_header()


def print_header():
    print('Legend       : S = new best R@0.5, C = new best CIDEr (each saves its checkpoint)')
    print('               cos_rand = cosine to a random caption (floor), txt_sim = similarity of')
    print('               chosen caption to GT caption (InternVideo2 text space), CIDEr = word overlap')
    print(RULE)
    widths = [sum(c[2] for c in cols) + len(cols) - 1 for _, cols in GROUPS]
    print(LEAD.format('', '', '') + ' | ' + ' | '.join(
        '{:^{}}'.format(name, w) for (name, _), w in zip(GROUPS, widths)))
    print(HEAD)
    print(RULE, flush=True)


def print_row(ep, t, lr, vals, mark):
    row = LEAD.format(ep, '%.0fs' % t, '%.1e' % lr) + ' | ' + _cells(
        lambda c: format(vals.get(c[1], float('nan')), '>%d%s' % (c[2], c[3])))
    print(row + ('  ' + mark if mark else ''), flush=True)
