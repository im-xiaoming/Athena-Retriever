"""Draw the architecture of the Athena (event segmentation + caption retrieval).

  python tools/draw_architecture.py   ->  docs/architecture.drawio.svg  and  docs/architecture.drawio

The .drawio.svg is a plain SVG for viewing with the same diagram embedded as draw.io XML (like
docs/Model Ar.drawio.svg); the .drawio file holds that XML alone. In draw.io, arrows are attached
to the boxes they start and end on, so they follow when a box is moved. Panels go from the
overview on the right to the details on the left, as in the original UniAV diagram.
Solid arrows run at inference (and training); dashed green arrows only during training.
Green tags name the loss that supervises a block.
"""
import html
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'docs', 'architecture.drawio.svg')
OUT_DRAWIO = os.path.join(ROOT, 'docs', 'architecture.drawio')

BG = '#1b1b1f'
TEXT = '#f2f2f2'
MUTED = '#c4c4cc'
BOX = '#111114'
WHITE = '#e6e6e6'
GREEN = '#7ddc7a'
KIND = {   # border colour per kind of block
    'same': WHITE,         # as in the original UniAV
    'new': '#ff9f1c',      # added in the Athena
    'changed': '#3ec5ff',  # kept, with different inputs or settings
    'frozen': '#b09cff',   # pretrained, frozen
    'train': GREEN,        # exists only during training
}
FONT = 'Helvetica, Arial, sans-serif'

cells, svg = [], []
BOXES = []   # every box, so arrows can be attached to the boxes they touch
_id = [1]


def nid():
    _id[0] += 1
    return 'c%d' % _id[0]


def esc(s):
    return html.escape(s, quote=True)


def text_lines(x, y, lines, size=14, color=TEXT, anchor='middle', weight='normal', lh=None):
    lh = lh or size * 1.3
    y0 = y - (len(lines) - 1) * lh / 2
    for i, l in enumerate(lines):
        svg.append('<text x="%.1f" y="%.1f" fill="%s" font-size="%d" font-weight="%s" text-anchor="%s" '
                   'dominant-baseline="middle" font-family="%s">%s</text>'
                   % (x, y0 + i * lh, color, size, weight, anchor, FONT, esc(l)))


def vertex(label, style, x, y, w, h):
    i = nid()
    cells.append('<mxCell id="%s" value="%s" style="%s" vertex="1" parent="1"><mxGeometry x="%d" y="%d" '
                 'width="%d" height="%d" as="geometry"/></mxCell>'
                 % (i, esc('<br>'.join(esc(l) for l in label.split('\n'))), style, x, y, w, h))
    return i


def rect(x, y, w, h, fill, stroke, rx=8, sw=1.5, arc=None):
    """Plain rectangle without text, in both the SVG and the draw.io model."""
    svg.append('<rect x="%d" y="%d" width="%d" height="%d" rx="%d" fill="%s" stroke="%s" stroke-width="%s"/>'
               % (x, y, w, h, rx, fill, stroke, sw))
    vertex('', 'rounded=1;%shtml=1;fillColor=%s;strokeColor=%s;strokeWidth=%s;'
           % ('arcSize=%d;' % arc if arc else '', fill, stroke, sw), x, y, w, h)


class Box:
    def __init__(self, x, y, w, h, label, kind='same', size=14, dashed=False, bold=False):
        self.x, self.y, self.w, self.h = x, y, w, h
        stroke = KIND[kind]
        svg.append('<rect x="%d" y="%d" width="%d" height="%d" rx="8" fill="%s" stroke="%s" stroke-width="%s"%s/>'
                   % (x, y, w, h, BOX, stroke, 1.4 if kind == 'same' else 2.2,
                      ' stroke-dasharray="8 5"' if dashed else ''))
        text_lines(x + w / 2, y + h / 2, label.split('\n'), size=size, weight='bold' if bold else 'normal')
        self.id = vertex(label, 'rounded=1;whiteSpace=wrap;html=1;fillColor=%s;strokeColor=%s;fontColor=%s;'
                         'fontSize=%d;strokeWidth=%s;%s%s'
                         % (BOX, stroke, TEXT, size, 1.4 if kind == 'same' else 2.2, 'dashed=1;' if dashed else '',
                            'fontStyle=1;' if bold else ''), x, y, w, h)
        BOXES.append(self)

    top = property(lambda s: (s.x + s.w / 2, s.y))
    bottom = property(lambda s: (s.x + s.w / 2, s.y + s.h))
    left = property(lambda s: (s.x, s.y + s.h / 2))
    right = property(lambda s: (s.x + s.w, s.y + s.h / 2))

    def at(self, fx, side='bottom'):
        """Point on the top or bottom edge at fraction fx of the width."""
        return (self.x + fx * self.w, self.y + (self.h if side == 'bottom' else 0))


def tag(box, label, below=False):
    """Green loss tag on the top-right corner of a box, or centred under it."""
    w = 8.5 * len(label) + 20
    if below:
        x, y = box.x + box.w / 2 - w / 2, box.y + box.h + 8
    else:
        x, y = box.x + box.w - w - 8, box.y - 13
    svg.append('<rect x="%.1f" y="%.1f" width="%.1f" height="24" rx="12" fill="#14301a" stroke="%s" stroke-width="1.5"/>'
               % (x, y, w, GREEN))
    text_lines(x + w / 2, y + 12, [label], size=12, color=GREEN, weight='bold')
    vertex(label, 'rounded=1;arcSize=50;html=1;fillColor=#14301a;strokeColor=%s;fontColor=%s;fontSize=12;fontStyle=1;'
           % (GREEN, GREEN), x, y, w, 24)


def panel(x, y, w, h, title, color, subtitle=None):
    svg.append('<rect x="%d" y="%d" width="%d" height="%d" fill="%s" stroke="#55555c" stroke-width="1"/>'
               % (x, y, w, h, color))
    svg.append('<text x="%d" y="%d" fill="%s" font-size="26" font-weight="bold" font-family="%s">%s</text>'
               % (x + 16, y + 36, TEXT, FONT, esc(title)))
    if subtitle:
        svg.append('<text x="%d" y="%d" fill="%s" font-size="15" font-family="%s">%s</text>'
                   % (x + 16, y + 60, MUTED, FONT, esc(subtitle)))
    vertex(title + ('\n' + subtitle if subtitle else ''),
           'rounded=0;whiteSpace=wrap;html=1;fillColor=%s;strokeColor=#55555c;fontColor=%s;fontSize=24;fontStyle=1;'
           'align=left;verticalAlign=top;spacingLeft=14;' % (color, TEXT), x, y, w, h)


def note(x, y, lines, size=14, color=MUTED, anchor='start', w=300, weight='normal'):
    text_lines(x, y, lines, size=size, color=color, anchor=anchor, weight=weight)
    lh = size * 1.3
    vertex('\n'.join(lines), 'text;html=1;fontColor=%s;fontSize=%d;align=%s;verticalAlign=middle;whiteSpace=wrap;'
           % (color, size, {'start': 'left', 'middle': 'center', 'end': 'right'}[anchor]),
           x - (w / 2 if anchor == 'middle' else (w if anchor == 'end' else 0)), y - len(lines) * lh / 2,
           w, len(lines) * lh)


def arrow(pts, label=None, train=False, lpos=0.5, lside='right', color=None):
    """Polyline with an arrow head; train=True draws a dashed green training-only path."""
    color = color or (GREEN if train else WHITE)
    d = 'M %.1f %.1f ' % pts[0] + ' '.join('L %.1f %.1f' % p for p in pts[1:])
    svg.append('<path d="%s" fill="none" stroke="%s" stroke-width="%s"%s marker-end="url(#%s)"/>'
               % (d, color, 1.8 if train else 1.5, ' stroke-dasharray="8 5"' if train else '',
                  'arrg' if train else 'arr'))
    if label:
        i = max(0, min(len(pts) - 2, int((len(pts) - 1) * lpos)))
        (x1, y1), (x2, y2) = pts[i], pts[i + 1]
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        lc = GREEN if train else MUTED
        if abs(x1 - x2) < 1:
            text_lines(mx + (9 if lside == 'right' else -9), my, label.split('\n'), size=13, color=lc,
                       anchor='start' if lside == 'right' else 'end')
        else:
            text_lines(mx, my - 11, label.split('\n'), size=13, color=lc)
    pts_xml = ''.join('<mxPoint x="%.1f" y="%.1f"/>' % p for p in pts[1:-1])
    # attach both ends to the boxes they lie on, at the same spot of the border
    ends, attrs = '', ''
    for (px, py), role, pre in ((pts[0], 'source', 'exit'), (pts[-1], 'target', 'entry')):
        b = on_box(px, py)
        if b is not None:
            attrs += ' %s="%s"' % (role, b.id)
            ends += '%sX=%.3f;%sY=%.3f;%sDx=0;%sDy=0;' % (pre, (px - b.x) / b.w, pre, (py - b.y) / b.h, pre, pre)
    cells.append('<mxCell id="%s" value="%s" style="html=1;endArrow=block;endFill=1;strokeColor=%s;fontColor=%s;'
                 'fontSize=13;rounded=0;edgeStyle=none;labelBackgroundColor=%s;%s%s" edge="1" parent="1"%s><mxGeometry relative="1" '
                 'as="geometry"><mxPoint x="%.1f" y="%.1f" as="sourcePoint"/><mxPoint x="%.1f" y="%.1f" '
                 'as="targetPoint"/><Array as="points">%s</Array></mxGeometry></mxCell>'
                 % (nid(), esc((label or '').replace('\n', '<br>')), color, GREEN if train else MUTED,
                    panel_colour(pts[0][0]), 'dashed=1;strokeWidth=1.8;' if train else 'strokeWidth=1.5;', ends, attrs,
                    pts[0][0], pts[0][1], pts[-1][0], pts[-1][1], pts_xml))


def panel_colour(x):
    """Background of the panel containing x, so edge labels in draw.io sit on the panel colour."""
    for px, pw, _, colour, _ in PANELS:
        if px <= x < px + pw:
            return colour
    return BG


def on_box(px, py, tol=2.0):
    """The most recent box whose border passes through (px, py), or None."""
    for b in reversed(BOXES):
        inside = b.x - tol <= px <= b.x + b.w + tol and b.y - tol <= py <= b.y + b.h + tol
        border = min(abs(px - b.x), abs(px - b.x - b.w), abs(py - b.y), abs(py - b.y - b.h)) <= tol
        if inside and border:
            return b
    return None


def down(a, b, label=None, src=None, dst=None, **kw):
    """Arrow from the bottom of a to the top of b, with one elbow when they are not aligned."""
    (x1, y1), (x2, y2) = src or a.bottom, dst or b.top
    if abs(x1 - x2) < 1:
        arrow([(x1, y1), (x2, y2)], label, **kw)
    else:
        ym = (y1 + y2) / 2
        arrow([(x1, y1), (x1, ym), (x2, ym), (x2, y2)], label, **kw)


def stack(x, y, w, labels, kind='same', h=36, gap=22, size=14):
    out = []
    for l in labels:
        b = Box(x, y + len(out) * (h + gap), w, h, l, kind, size=size)
        if out:
            down(out[-1], b)
        out.append(b)
    return out


# ============================================================================== layout
W, H = 3330, 1600
PANELS = [(0, 570, 'ConvTransformerBackbone', '#1f2a36', 'giữ nguyên như UniAV'),
          (580, 640, 'Prediction Heads', '#33240f', 'chạy trên mọi mức FPN, dùng chung trọng số'),
          (1230, 830, 'Segment vector & caption choice', '#2a2338', 'SegmentContext + chọn câu trong kho caption'),
          (2070, 1260, 'Athena: overview', '#2b1f1f', 'nét liền = suy luận (và cả lúc train);  '
                                                       'nét đứt xanh lá = chỉ lúc train')]
for x, w, t, c, s in PANELS:
    panel(x, 0, w, H, t, c, s)

# ------------------------------------------------------------------ overview (right)
x0 = 2070
cx = x0 + 445
xr = x0 + 945          # right column: captions and the teacher
wr = 295
vid = Box(cx - 340, 95, 290, 50, 'Video frames\n2 fps, 224 x 224')
aud = Box(cx + 50, 95, 290, 50, 'Audio track\n16 kHz mono')
iv2 = Box(cx - 355, 190, 320, 72, 'InternVideo2-1B vision\n(frozen)\n4 frames per 1 s window', 'frozen', bold=True)
bts = Box(cx + 35, 190, 320, 72, 'BEATs audio encoder\n(frozen)\n3 s window per second', 'frozen', bold=True)
down(vid, iv2); down(aud, bts)
nv = Box(cx - 355, 320, 320, 42, 'L2 norm, resample to 256 steps', 'new')
na = Box(cx + 35, 320, 320, 42, 'L2 norm, resample to 256 steps', 'new')
down(iv2, nv, 'v768: 1 vector / s', lside='left'); down(bts, na, 'a768: 1 vector / s')
bb = Box(cx - 310, 430, 620, 66, 'ConvTransformerBackbone  (unchanged)\nV <-> A cross-attention, 6 FPN levels', 'changed',
         size=16, bold=True)
down(nv, bb, 'input_V (B, 768, 256)', lside='left', lpos=0.0)
down(na, bb, 'input_A (B, 768, 256)', lpos=0.0)
cat = Box(cx - 250, 545, 500, 42, "Cat(V, A) on every level  ->  (B, 1024, T')")
down(bb, cat)
he = Box(cx - 415, 640, 260, 84, 'Event Head\nis an event\nhappening here?', 'new', bold=True)
hb = Box(cx - 130, 640, 260, 84, 'Boundary Head\ndistance to start / end\n+ IoU quality', 'changed', bold=True)
hm = Box(cx + 155, 640, 260, 84, 'Embed Head\none 512-d vector\nper step', 'new', bold=True)
for b in (he, hb, hm):
    down(cat, b)
tag(he, 'L_event'); tag(hb, 'L_reg + L_iou'); tag(hm, 'L_emb')
sc = Box(cx - 415, 790, 545, 64, 'score = event^0.7 x IoU^0.3,  Soft-NMS\n-> predicted segments (t_s, t_e)', 'new')
down(he, sc); down(hb, sc)
gt = Box(cx + 155, 790, 160, 64, 'GT segments\n+ copy shifted\nby up to 20%', 'train', dashed=True, size=13)
sctx = Box(cx - 415, 935, 830, 64, 'SegmentContext:  span mean + whole-video mean + position\n->  segment vector q (512)',
           'new', size=16, bold=True)
down(sc, sctx, 'inference:\npredicted segments', lside='left', lpos=0.0)
down(gt, sctx, 'training', train=True, dst=(gt.x + gt.w / 2, sctx.y))
arrow([(cx + 365, hm.y + hm.h), (cx + 365, sctx.y)], 'level 0', lpos=0.0)
tag(sctx, 'L_span + L_omni_av + L_omni_txt')
ret = Box(cx - 415, 1075, 830, 70, 'Caption retrieval: for each of the 8218 train captions\n'
          'score = cos(q, caption vector)   [+ teacher score, optional]', 'new', size=15, bold=True)
down(sctx, ret, 'q')
mbr = Box(cx - 415, 1195, 830, 50, 'Pick by consensus (MBR) among the top 20 captions', 'new', size=15)
down(ret, mbr)
out = Box(cx - 330, 1290, 660, 50, 'Output: every event (t_s, t_e) with one caption', size=16, bold=True)
down(mbr, out)

# right column: teacher (top) and caption vectors (bottom)
tin = Box(xr, 95, wr, 56, 'train GT clips (cut, video + audio)\nand their captions', 'train', dashed=True, size=13)
tch = Box(xr, 190, wr, 72, 'OmniRetriever-7B teacher\n(frozen, run once offline)', 'frozen', bold=True)
down(tin, tch, train=True)
tav = Box(xr, 320, wr, 56, 'audio+video vector\nof each GT clip (3584)', 'train', dashed=True, size=13)
ttx = Box(xr, 430, wr, 56, 'text vector of each\ntrain caption (8218 x 3584)', 'frozen', size=13)
down(tch, tav, train=True)
arrow([(xr + wr - 25, tch.y + tch.h), (xr + wr - 25, ttx.y)])
# av target -> SegmentContext tag (training only), along the gap left of the right column
arrow([tav.left, (xr - 40, tav.left[1]), (xr - 40, sctx.y + 32), (sctx.x + sctx.w, sctx.y + 32)],
      train=True)
note(xr - 50, 560, ['target of q', '(L_omni_av)'], size=13, color=GREEN, anchor='end', w=150)
caps = Box(xr, 650, wr, 56, 'train captions\n8218 unique sentences')
opt = Box(xr, 760, wr, 56, 'InternVideo2 text encoder\n(frozen)  ->  512', 'frozen', size=13)
cpj = Box(xr, 870, wr, 56, 'clip_proj: Linear 512 -> 512\n(trained)', 'new', size=13)
down(caps, opt); down(opt, cpj)
arrow([(cpj.x + 150, cpj.y + cpj.h), (cpj.x + 150, ret.y + 20), (ret.x + ret.w, ret.y + 20)],
      'caption vectors', lpos=1.0)
arrow([ttx.right, (xr + wr + 14, ttx.right[1]), (xr + wr + 14, ret.y + 54), (ret.x + ret.w, ret.y + 54)],
      'teacher text vectors\n(optional)', lpos=1.0, color='#8c8c94')

# loss summary and legend
note(x0 + 20, 1385, ['Loss (trọng số)'], size=15, color=TEXT, weight='bold', w=200)
for i, (n_, d_) in enumerate([
        ('L_event', 'focal: có sự kiện ở bước này không (1.0)'),
        ('L_reg', 'centre-DIoU cho khoảng cách tới đầu / cuối (1.0)'),
        ('L_iou', 'BCE giữa IoU quality và IoU thật (1.0)'),
        ('L_emb, L_span', 'cross-entropy trên CẢ kho caption, nhãn mềm (0.2)'),
        ('L_omni_txt', 'như trên, với vector text của thầy (0.2)'),
        ('L_omni_av', 'q phải nhận ra đúng clip GT của nó (0.2)')]):
    note(x0 + 20, 1415 + i * 27, [n_], size=14, color=GREEN, weight='bold', w=130)
    note(x0 + 150, 1415 + i * 27, [d_], size=14, color=TEXT, w=520)
lx, ly = x0 + 720, 1370
rect(lx, ly, 520, 225, '#151518', '#55555c', sw=1)
note(lx + 18, ly + 22, ['Chú thích'], size=15, color=TEXT, weight='bold', w=200)
arrow([(lx + 20, ly + 55), (lx + 90, ly + 55)]); note(lx + 105, ly + 55, ['suy luận (và cả lúc train)'], w=300)
arrow([(lx + 20, ly + 85), (lx + 90, ly + 85)], train=True); note(lx + 105, ly + 85, ['chỉ lúc train'], w=300)
for i, (k, t) in enumerate([('same', 'như UniAV gốc'), ('changed', 'giữ, đổi đầu vào / cấu hình'),
                            ('new', 'mới thêm'), ('frozen', 'pretrained, đóng băng')]):
    yy = ly + 118 + (i // 2) * 34
    xx = lx + 20 + (i % 2) * 250
    rect(xx, yy - 11, 40, 22, BOX, KIND[k], rx=5, sw=2)
    note(xx + 50, yy, [t], size=13, w=200)
rect(lx + 20, ly + 187, 64, 22, '#14301a', GREEN, rx=11, arc=50)
note(lx + 94, ly + 198, ['loss dạy khối này'], size=13, w=300)

# ------------------------------------------------------------------ segment vector & caption choice
x0 = 1230
cx = x0 + 415
r0 = Box(cx - 385, 95, 360, 56, 'level-0 features of the Embed Head\n(512 x 256 steps)', 'new', size=13)
r1 = Box(cx + 25, 95, 360, 56, 'segment [s, e]\npredicted (inference) or GT (training)', 'new', size=13)
s1 = Box(cx - 395, 205, 250, 60, 'span mean\nwhat happens in [s, e]', 'new', size=13)
s2 = Box(cx - 125, 205, 250, 60, 'whole-video mean\nwhich dish is cooked', 'new', size=13)
s3 = Box(cx + 145, 205, 250, 60, 'position s/T, e/T\nearly or late step', 'new', size=13)
down(r0, s1, src=r0.at(0.3)); down(r0, s2, src=r0.at(0.7)); down(r1, s3); down(r1, s1, src=r1.at(0.15))
cc = Box(cx - 170, 320, 340, 38, 'concat  ->  512 + 512 + 2', 'new')
for s_ in (s1, s2, s3):
    down(s_, cc)
mlp = stack(cx - 170, 395, 340, ['Linear 1026 -> 512', 'GELU', 'Linear 512 -> 512 (starts at 0)'], 'new', h=34, gap=16)
down(cc, mlp[0])
add = Box(cx - 35, 560, 70, 38, '+', 'new', size=20, bold=True)
down(mlp[-1], add)
arrow([(s1.x + 30, s1.y + s1.h), (s1.x + 30, 579), (add.x, 579)], 'residual: q starts as the span mean', lpos=1.0)
q = Box(cx - 170, 640, 340, 42, 'L2 norm  ->  segment vector q', 'new', bold=True)
down(add, q)
tag(q, 'L_span')
c1 = Box(cx - 395, 740, 360, 60, 'cos(q, caption vector)\ncaption vector = clip_proj(InternVideo2 text)', 'new', size=13)
c2 = Box(cx + 35, 740, 360, 60, 'omni_proj: 512 -> 1024 -> 3584, L2\ncos with the teacher text vector', 'new', size=13)
down(q, c1); down(q, c2)
tag(c2, 'L_omni_*')
ss = Box(cx - 260, 860, 520, 40, 'score of each caption = cos in caption space  (+ teacher cos, optional)',
         'new', size=13)
down(c1, ss); down(c2, ss)
# worked example of the consensus pick
ex_y = 950
ex_box = Box(cx - 390, ex_y, 780, 250, '', 'new')
down(ss, None, dst=(cx, ex_y), src=ss.bottom)
note(cx - 370, ex_y + 26, ['Top 20 by score, then pick by consensus (example)'], size=15, color=TEXT,
     weight='bold', w=700)
cols_x = [cx - 370, cx + 90, cx + 210]
rows = [('candidate caption', 'score', 'agreement'),
        ('add oil to the pan', '0.62', '0.31'),
        ('cut the onion', '0.61', '0.52   <- picked'),
        ('slice the onions', '0.60', '0.50'),
        ('chop onion into pieces', '0.59', '0.49'),
        ('...  (20 candidates)', '', '')]
for i, r in enumerate(rows):
    yy = ex_y + 62 + i * 29
    for j, (xx, v) in enumerate(zip(cols_x, r)):
        bold = i == 0 or (i == 2)
        col = MUTED if i == 0 else (KIND['new'] if i == 2 else TEXT)
        note(xx, yy, [v], size=14, color=col, weight='bold' if bold else 'normal', w=260)
oc = Box(cx - 200, 1250, 400, 46, 'caption: "cut the onion"', 'same', size=16, bold=True)
down(None, oc, src=(cx, ex_y + 250))
note(x0 + 20, 1410, [
    'agreement = mức giống của một ứng viên với các ứng viên còn lại, có trọng số theo',
    'điểm của chúng. "add oil" đứng đầu nhưng đơn độc; ba câu về hành tây ủng hộ nhau,',
    'nên bước "cắt hành" mới là cách hiểu đáng tin (số liệu trong bảng là ví dụ minh hoạ).',
    '',
    'Lúc train: q của đoạn GT phải chấm câu đúng cao hơn 8217 câu còn lại (L_span);',
    'thêm một bản đoạn lệch biên tới 20% để q chịu được biên đoán hơi sai lúc suy luận.',
    'Lớp Linear cuối bắt đầu bằng 0, nên ban đầu q chính là trung bình đoạn.',
    '',
    'omni_proj chủ yếu để thầy dạy q lúc train. Cộng thêm điểm thầy lúc chọn câu không giúp',
    'đo được (chênh <= 0.003), nên athena chỉ dùng điểm trong không gian caption.'],
    size=14, color=TEXT, w=800)

# ------------------------------------------------------------------ heads
x0 = 580
cols = [x0 + 20, x0 + 230, x0 + 440]
cw = 180
titles = [('Event Head', 'new'), ('Boundary Head', 'changed'), ('Embed Head', 'new')]
towers = []
for c, (t, k) in zip(cols, titles):
    Box(c, 95, cw, 46, "x = Cat(V, A)\n(B, 1024, T')", size=13)
    text_lines(c + cw / 2, 172, [t], size=17, weight='bold', color=KIND[k])
    s = stack(c, 200, cw, ['Masked Conv 1D', 'Layer Norm', 'ReLU'], size=14)
    towers.append(s)
    note(c + cw + 5, 275, ['x2'], w=24, size=13)
e1 = Box(cols[0], 420, cw, 52, 'Masked Conv 1D -> 1\nstarts at p = 0.01', 'new', size=13)
down(towers[0][-1], e1)
e2 = Box(cols[0], 530, cw, 52, "event logit\n(B, T', 1)", 'new', size=13)
down(e1, e2)
tag(e2, 'L_event', below=True)
bx = cols[1]
b1 = Box(bx, 420, 84, 52, 'MConv\n-> 2', size=13)
b2 = Box(bx + 96, 420, 84, 52, 'MConv\n-> 1', 'new', size=13)
down(towers[1][-1], b1); down(towers[1][-1], b2)
b3 = Box(bx, 520, 84, 36, 'Scale_l', size=13)
b4 = Box(bx, 600, 84, 36, 'ReLU', size=13)
down(b1, b3); down(b3, b4)
b5 = Box(bx, 690, 84, 56, '(d_s, d_e)\ndistances', size=13)
down(b4, b5)
b6 = Box(bx + 96, 690, 84, 56, 'IoU\nquality', 'new', size=13)
down(b2, b6)
tag(b5, 'L_reg', below=True); tag(b6, 'L_iou', below=True)
mx = cols[2]
m1 = Box(mx, 420, cw, 52, 'Masked Conv 1D\nvis_proj (512)', 'new', size=13)
down(towers[2][-1], m1)
m2 = Box(mx, 530, 84, 52, 'L2 norm', 'new', size=13)
m3 = Box(mx + 96, 530, 84, 52, 'level 0\nonly', 'new', size=13)
down(m1, m2, src=m1.at(0.23)); down(m1, m3, src=m1.at(0.77))
m4 = Box(mx, 640, 84, 56, 'vector\nper step', 'new', size=13)
m5 = Box(mx + 96, 640, 84, 56, 'Segment\nContext', 'new', size=13)
down(m2, m4); down(m3, m5)
tag(m4, 'L_emb', below=True)
note(x0 + 20, 990, [
    'Event Head thay cho head phân lớp của UniAV gốc:',
    'chỉ 1 kênh "ở bước này có sự kiện không",',
    'không chia theo lớp.',
    '',
    'Boundary Head giữ nhánh hồi quy khoảng cách tới',
    'điểm đầu / cuối của UniAV, thêm nhánh tự đoán',
    'đoạn của mình khớp GT tới đâu (IoU). Lúc suy luận,',
    'IoU này dùng để xếp hạng các đoạn.',
    '',
    'Embed Head cho mỗi bước một vector trong không',
    'gian caption. Đầu ra ở mức 0 (trước L2 norm) được',
    'SegmentContext lấy trung bình theo đoạn.'], size=14, color=TEXT, w=600)

# ------------------------------------------------------------------ backbone
x0 = 0
cx = 285
xv = Box(cx - 245, 95, 200, 40, 'x_V (B, 768, T)', 'changed', size=13)
xa = Box(cx + 45, 95, 200, 40, 'x_A (B, 768, T)', 'changed', size=13)
ev_ = stack(cx - 245, 170, 200, ['Masked Conv 1D', 'Layer Norm', 'ReLU'], size=13, h=34, gap=18)
ea_ = stack(cx + 45, 170, 200, ['Masked Conv 1D', 'Layer Norm', 'ReLU'], size=13, h=34, gap=18)
down(xv, ev_[0]); down(xa, ea_[0])
note(cx - 40, 238, ['x2'], w=24, size=13); note(cx + 250, 238, ['x2'], w=24, size=13)
pe = Box(cx - 150, 370, 300, 42, 'Sinusoid Encoding', size=14)
down(ev_[-1], pe); down(ea_[-1], pe)
sv = Box(cx - 265, 470, 230, 52, 'Self-Transformer (V)\ns = (1, 1)', size=13)
sa = Box(cx + 35, 470, 230, 52, 'Self-Transformer (A)\ns = (1, 1)', size=13)
down(pe, sv); down(pe, sa)
note(cx + 268, 496, ['x2'], w=24, size=13)
cv = Box(cx - 265, 610, 230, 64, 'Cross-Transformer (Va)\ns = (1, 1)\nV = (k, v)', size=13)
ca = Box(cx + 35, 610, 230, 64, 'Cross-Transformer (Av)\ns = (1, 1)\nA = (k, v)', size=13)
arrow([sv.bottom, cv.top]); arrow([sa.bottom, ca.top])
arrow([sv.bottom, (sv.bottom[0], 562), (ca.x + 40, 562), (ca.x + 40, ca.y)])
arrow([sa.bottom, (sa.bottom[0], 576), (cv.x + cv.w - 40, 576), (cv.x + cv.w - 40, cv.y)])
dv = Box(cx - 265, 770, 230, 64, 'Cross-Transformer (Va)\nlevels 1-5, stride 2\nV = (k, v)', size=13)
da = Box(cx + 35, 770, 230, 64, 'Cross-Transformer (Av)\nlevels 1-5, stride 2\nA = (k, v)', size=13)
arrow([cv.bottom, dv.top]); arrow([ca.bottom, da.top])
arrow([cv.bottom, (cv.bottom[0], 715), (da.x + 40, 715), (da.x + 40, da.y)])
arrow([ca.bottom, (ca.bottom[0], 729), (dv.x + dv.w - 40, 729), (dv.x + dv.w - 40, dv.y)])
note(cx + 268, 802, ['x5'], w=24, size=13)
ov = Box(cx - 265, 890, 230, 46, "V: 6 levels (B, 512, T')", size=13)
oa = Box(cx + 35, 890, 230, 46, "A: 6 levels (B, 512, T')", size=13)
down(dv, ov); down(da, oa)
note(x0 + 20, 1040, [
    'Giống hệt UniAV gốc (TransformersBlock và MaskedMHCA',
    'xem docs/Model Ar.drawio.svg).',
    '',
    'Chỉ đổi số kênh vào: 768 từ InternVideo2 / BEATs',
    'thay cho 1536 từ ONE-PEACE.',
    '',
    "Các mức FPN: T' = 256, 128, 64, 32, 16, 8;  C = 512."], size=14, color=TEXT, w=540)

# ============================================================================== write
mx_xml = ('<mxfile host="tools/draw_architecture.py"><diagram name="Athena" id="athena"><mxGraphModel dx="%d" '
          'dy="%d" grid="0" gridSize="10" page="0" background="%s"><root><mxCell id="0"/><mxCell id="1" parent="0"/>%s'
          '</root></mxGraphModel></diagram></mxfile>' % (W, H, BG, ''.join(cells)))
doc = ('<?xml version="1.0" encoding="UTF-8"?>\n'
       '<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="%dpx" height="%dpx" viewBox="0 0 %d %d" '
       'content="%s">\n<defs>'
       '<marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
       '<path d="M 0 0 L 10 5 L 0 10 z" fill="%s"/></marker>'
       '<marker id="arrg" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
       '<path d="M 0 0 L 10 5 L 0 10 z" fill="%s"/></marker></defs>\n'
       '<rect width="100%%" height="100%%" fill="%s"/>\n%s\n</svg>\n'
       % (W, H, W, H, esc(mx_xml), WHITE, GREEN, BG, '\n'.join(svg)))
with open(OUT, 'w') as f:
    f.write(doc)
with open(OUT_DRAWIO, 'w') as f:
    f.write(mx_xml + '\n')
print('wrote %s and %s (%d cells)' % (OUT, OUT_DRAWIO, len(cells)))
