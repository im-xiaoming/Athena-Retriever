"""Draw the architecture of the new UniAV (event segmentation + caption retrieval).

  python tools/draw_architecture.py      ->  docs/UniAV_new.drawio.svg

Writes one .drawio.svg: a plain SVG for viewing, with the same diagram embedded as draw.io
XML so it opens editable in draw.io (like docs/Model Ar.drawio.svg). Panels go from the
overview on the right to the details on the left, as in the original UniAV diagram.
"""
import html
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'docs', 'UniAV_new.drawio.svg')

BG = '#1b1b1f'
TEXT = '#f2f2f2'
MUTED = '#b8b8c0'
BOX = '#111114'
KIND = {   # border colour per kind of block
    'same': '#e6e6e6',     # as in the original UniAV
    'new': '#ff9f1c',      # added in the new UniAV
    'changed': '#3ec5ff',  # kept, but with different inputs or settings
    'frozen': '#9b8cff',   # pretrained, frozen
    'loss': '#7ddc7a',
}
PANEL = ['#22301f', '#2a2338', '#33240f', '#1f2a36', '#2b1f1f']

cells, svg = [], []
_id = [1]


def nid():
    _id[0] += 1
    return 'c%d' % _id[0]


def esc(s):
    return html.escape(s, quote=True)


def text_lines(x, y, lines, size=13, color=TEXT, anchor='middle', weight='normal', lh=None):
    lh = lh or size * 1.3
    y0 = y - (len(lines) - 1) * lh / 2
    for i, l in enumerate(lines):
        svg.append('<text x="%.1f" y="%.1f" fill="%s" font-size="%d" font-weight="%s" text-anchor="%s" '
                   'dominant-baseline="middle" font-family="Helvetica, Arial, sans-serif">%s</text>'
                   % (x, y0 + i * lh, color, size, weight, anchor, esc(l)))


class Box:
    def __init__(self, x, y, w, h, label, kind='same', size=12, dashed=False, bold=False):
        self.x, self.y, self.w, self.h, self.id = x, y, w, h, nid()
        stroke = KIND[kind]
        lines = label.split('\n')
        svg.append('<rect x="%d" y="%d" width="%d" height="%d" rx="7" fill="%s" stroke="%s" stroke-width="%s"%s/>'
                   % (x, y, w, h, BOX, stroke, 2 if kind != 'same' else 1.3,
                      ' stroke-dasharray="7 5"' if dashed else ''))
        text_lines(x + w / 2, y + h / 2, lines, size=size, weight='bold' if bold else 'normal')
        style = ('rounded=1;whiteSpace=wrap;html=1;fillColor=%s;strokeColor=%s;fontColor=%s;fontSize=%d;%s%s'
                 % (BOX, stroke, TEXT, size, 'dashed=1;' if dashed else '', 'fontStyle=1;' if bold else ''))
        cells.append('<mxCell id="%s" value="%s" style="%s" vertex="1" parent="1"><mxGeometry x="%d" y="%d" '
                     'width="%d" height="%d" as="geometry"/></mxCell>'
                     % (self.id, esc('<br>'.join(esc(l) for l in lines)), style, x, y, w, h))

    @property
    def top(self):
        return (self.x + self.w / 2, self.y)

    @property
    def bottom(self):
        return (self.x + self.w / 2, self.y + self.h)

    @property
    def left(self):
        return (self.x, self.y + self.h / 2)

    @property
    def right(self):
        return (self.x + self.w, self.y + self.h / 2)


def panel(x, y, w, h, title, color):
    svg.append('<rect x="%d" y="%d" width="%d" height="%d" fill="%s" stroke="#55555c" stroke-width="1"/>'
               % (x, y, w, h, color))
    svg.append('<text x="%d" y="%d" fill="%s" font-size="22" font-weight="bold" '
               'font-family="Helvetica, Arial, sans-serif">%s</text>' % (x + 14, y + 32, TEXT, esc(title)))
    cells.append('<mxCell id="%s" value="%s" style="rounded=0;whiteSpace=wrap;html=1;fillColor=%s;strokeColor=#55555c;'
                 'fontColor=%s;fontSize=22;fontStyle=1;align=left;verticalAlign=top;spacingLeft=12;" vertex="1" '
                 'parent="1"><mxGeometry x="%d" y="%d" width="%d" height="%d" as="geometry"/></mxCell>'
                 % (nid(), esc(title), color, TEXT, x, y, w, h))


def note(x, y, lines, size=12, color=MUTED, anchor='start', w=300):
    text_lines(x, y, lines, size=size, color=color, anchor=anchor)
    lh = size * 1.3
    cells.append('<mxCell id="%s" value="%s" style="text;html=1;fontColor=%s;fontSize=%d;align=%s;verticalAlign=middle;'
                 'whiteSpace=wrap;" vertex="1" parent="1"><mxGeometry x="%d" y="%d" width="%d" height="%d" '
                 'as="geometry"/></mxCell>'
                 % (nid(), esc('<br>'.join(esc(l) for l in lines)), color, size,
                    {'start': 'left', 'middle': 'center', 'end': 'right'}[anchor],
                    x - (w / 2 if anchor == 'middle' else (w if anchor == 'end' else 0)),
                    y - len(lines) * lh / 2, w, len(lines) * lh))


def arrow(pts, label=None, src=None, dst=None, dashed=False, color='#e6e6e6', lpos=0.5, lside='right'):
    """Polyline through pts with an arrow head at the end; optional label beside the middle."""
    d = 'M %.1f %.1f ' % pts[0] + ' '.join('L %.1f %.1f' % p for p in pts[1:])
    svg.append('<path d="%s" fill="none" stroke="%s" stroke-width="1.4"%s marker-end="url(#arr)"/>'
               % (d, color, ' stroke-dasharray="6 4"' if dashed else ''))
    if label:
        i = max(0, min(len(pts) - 2, int((len(pts) - 1) * lpos)))
        (x1, y1), (x2, y2) = pts[i], pts[i + 1]
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        if abs(x1 - x2) < 1:   # vertical segment: label to the side
            text_lines(mx + (8 if lside == 'right' else -8), my, label.split('\n'), size=11, color=MUTED,
                       anchor='start' if lside == 'right' else 'end')
        else:
            text_lines(mx, my - 10, label.split('\n'), size=11, color=MUTED)
    pts_xml = ''.join('<mxPoint x="%.1f" y="%.1f"/>' % p for p in pts[1:-1])
    style = 'html=1;endArrow=block;endFill=1;strokeColor=%s;fontColor=%s;fontSize=11;rounded=0;%s' % (
        color, MUTED, 'dashed=1;' if dashed else '')
    cells.append('<mxCell id="%s" value="%s" style="%s" edge="1" parent="1"><mxGeometry relative="1" as="geometry">'
                 '<mxPoint x="%.1f" y="%.1f" as="sourcePoint"/><mxPoint x="%.1f" y="%.1f" as="targetPoint"/>'
                 '<Array as="points">%s</Array></mxGeometry></mxCell>'
                 % (nid(), esc((label or '').replace('\n', '<br>')), style, pts[0][0], pts[0][1],
                    pts[-1][0], pts[-1][1], pts_xml))


def down(a, b, label=None, **kw):
    """Arrow from the bottom of a to the top of b, with one elbow if they are not aligned."""
    (x1, y1), (x2, y2) = a.bottom, b.top
    if abs(x1 - x2) < 1:
        arrow([(x1, y1), (x2, y2)], label, **kw)
    else:
        ym = (y1 + y2) / 2
        arrow([(x1, y1), (x1, ym), (x2, ym), (x2, y2)], label, **kw)


def stack(x, y, w, labels, kind='same', h=34, gap=20, size=12):
    """Vertical chain of boxes joined by arrows; returns the boxes."""
    out = []
    for i, l in enumerate(labels):
        b = Box(x, y + i * (h + gap), w, h, l, kind, size=size)
        if out:
            down(out[-1], b)
        out.append(b)
    return out


# ============================================================================== layout
W, H = 3720, 1400
PX = [0, 690, 1420, 2080, 2780]    # panel x
PW = [680, 720, 650, 690, 940]     # panel widths

# ------------------------------------------------------------------ panel 5: overview (right)
x0 = PX[4]
panel(x0, 0, PW[4], H, 'New UniAV: overview', PANEL[4])
cx = x0 + PW[4] / 2
vid = Box(cx - 300, 60, 230, 40, 'Video frames\n2 fps, 224 x 224', 'same')
aud = Box(cx + 70, 60, 230, 40, 'Audio\n16 kHz mono', 'same')
iv2 = Box(cx - 330, 135, 290, 62, 'InternVideo2-1B Stage2 vision\n4 frames per 1 s window\n(attention-pooled)', 'frozen', bold=True)
bts = Box(cx + 40, 135, 290, 62, 'BEATs audio encoder\n(from InternVideo2-6B)\n3 s window per second', 'frozen', bold=True)
down(vid, iv2); down(aud, bts)
nv = Box(cx - 330, 240, 290, 40, 'L2 norm + resample to T = 256', 'new')
na = Box(cx + 40, 240, 290, 40, 'L2 norm + resample to T = 256', 'new')
down(iv2, nv, 'v768  (opt. + v512 text-aligned)\n1 row / second')
down(bts, na, 'a768\n1 row / second')
op = Box(x0 + 15, 396, 200, 58, 'ONE-PEACE V / A\n(1536 each), optional:\nconcat on channels', 'changed', dashed=True, size=11)
bb = Box(cx - 230, 395, 460, 60, 'ConvTransformerBackbone\n(unchanged from UniAV, input C_in = 768)', 'changed', size=14, bold=True)
down(nv, bb, 'input_V (B, 768, 256)', lside='left', lpos=0.0)
down(na, bb, 'input_A (B, 768, 256)', lpos=0.0)
arrow([op.right, (bb.x, op.right[1])], dashed=True, color=KIND['changed'])
note(cx + 250, 425, ['6 FPN levels', "T' = 256, 128, 64, 32, 16, 8", 'C = 512 per modality'], w=200)
cat = Box(cx - 160, 495, 320, 40, "Cat(V, A) at every FPN level -> C = 1024", 'same')
down(bb, cat)
he = Box(cx - 340, 585, 200, 52, 'Event Head\nevent / no event', 'new', bold=True)
hb = Box(cx - 100, 585, 200, 52, 'Boundary Head\n(d_s, d_e) + IoU quality', 'changed', bold=True)
hm = Box(cx + 140, 585, 200, 52, 'Embed Head\nvector per step (512)', 'new', bold=True)
for h_ in (he, hb, hm):
    down(cat, h_)
note(x0 + 15, 611, ['replaces the', 'class head'], w=110)
sc = Box(cx - 340, 690, 440, 52, 'score = sigmoid(event)^0.7 * sigmoid(IoU)^0.3\nSoft-NMS (class agnostic)', 'new')
down(he, sc); down(hb, sc)
sctx = Box(cx - 340, 800, 440, 52, 'SegmentContext\nspan mean + whole-video mean + position', 'new', bold=True)
down(sc, sctx, 'segments (t_s, t_e)')
arrow([hm.bottom, (hm.bottom[0], 826), (sctx.x + sctx.w, 826)], 'level-0 features', lpos=0.0)
ret = Box(cx - 340, 905, 440, 66, 'Caption retrieval\ncosine to 8218 train captions\n(caption space + OmniRetriever space)', 'new', bold=True)
down(sctx, ret, 'segment vector q (512)')
mbr = Box(cx - 340, 1025, 440, 52, 'Consensus pick (MBR) among the top 20', 'new')
down(ret, mbr)
outb = Box(cx - 340, 1125, 440, 44, 'out: (t_s, t_e, caption) for every event', 'same', bold=True)
down(mbr, outb)
note(x0 + 20, 1235, [
    'Từ phải sang trái: tổng quan -> backbone -> các head -> retrieval -> loss.',
    'Viền: trắng = như UniAV gốc, xanh dương = giữ nhưng đổi đầu vào/cấu hình,',
    'cam = mới thêm, tím = encoder pretrained đóng băng, xanh lá = loss lúc train.',
    'Khác UniAV gốc: bỏ head phân lớp (class-aware), mỗi đoạn được gán một câu',
    'chọn từ kho caption train thay vì một nhãn lớp; encoder ONE-PEACE thay bằng',
    'InternVideo2 + BEATs (ONE-PEACE còn là tuỳ chọn ghép kênh).'], size=13, color=TEXT, w=900)

# ------------------------------------------------------------------ panel 4: backbone (unchanged)
x0 = PX[3]
panel(x0, 0, PW[3], H, 'ConvTransformerBackbone (unchanged)', PANEL[3])
cx = x0 + PW[3] / 2
xv = Box(cx - 230, 70, 140, 34, 'x_V (B, 768, T)', 'changed')
xa = Box(cx + 90, 70, 140, 34, 'x_A (B, 768, T)', 'changed')
ev_ = stack(cx - 250, 135, 180, ['Masked Conv 1D', 'Layer Norm', 'ReLU'])
ea_ = stack(cx + 70, 135, 180, ['Masked Conv 1D', 'Layer Norm', 'ReLU'])
down(xv, ev_[0]); down(xa, ea_[0])
note(cx - 60, 200, ['x2'], w=30); note(cx + 260, 200, ['x2'], w=30)
pe = Box(cx - 150, 340, 300, 44, 'Sinusoid Encoding', 'same')
down(ev_[-1], pe); down(ea_[-1], pe)
sv = Box(cx - 290, 440, 230, 50, 'Self-Transformer Block (V)\ns = (1, 1)')
sa = Box(cx + 60, 440, 230, 50, 'Self-Transformer Block (A)\ns = (1, 1)')
down(pe, sv); down(pe, sa)
note(cx + 300, 465, ['x2'], w=30)
cv = Box(cx - 290, 570, 230, 62, 'Cross-Transformer (Va)\ns = (1, 1)\nV = (k, v)')
ca = Box(cx + 60, 570, 230, 62, 'Cross-Transformer (Av)\ns = (1, 1)\nA = (k, v)')
arrow([sv.bottom, cv.top]); arrow([sa.bottom, ca.top])
arrow([sv.bottom, (sv.bottom[0], 530), (ca.top[0] - 40, 530), (ca.top[0] - 40, ca.y)])
arrow([sa.bottom, (sa.bottom[0], 540), (cv.top[0] + 40, 540), (cv.top[0] + 40, cv.y)])
dv = Box(cx - 290, 720, 230, 62, 'Cross-Transformer (Va)\nl = 0, 1, 2, 3, 4  (stride 2)\nV = (k, v)')
da = Box(cx + 60, 720, 230, 62, 'Cross-Transformer (Av)\nl = 0, 1, 2, 3, 4  (stride 2)\nA = (k, v)')
arrow([cv.bottom, dv.top]); arrow([ca.bottom, da.top])
arrow([cv.bottom, (cv.bottom[0], 670), (da.top[0] - 40, 670), (da.top[0] - 40, da.y)])
arrow([ca.bottom, (ca.bottom[0], 680), (dv.top[0] + 40, 680), (dv.top[0] + 40, dv.y)])
note(cx + 300, 751, ['x5'], w=30)
ov = Box(cx - 290, 840, 230, 44, "V levels: (B, 512, T')", 'same')
oa = Box(cx + 60, 840, 230, 44, "A levels: (B, 512, T')", 'same')
down(dv, ov); down(da, oa)
note(x0 + 20, 960, [
    'Giống hệt UniAV gốc (TransformersBlock và MaskedMHCA',
    'xem docs/Model Ar.drawio.svg). Chỉ đổi số kênh vào:',
    '768 (InternVideo2) thay cho 1536 (ONE-PEACE),',
    'hoặc 2304 khi ghép cả hai.',
    '',
    "6 mức FPN: T' = 256, 128, 64, 32, 16, 8; C = 512."], size=13, color=TEXT, w=640)

# ------------------------------------------------------------------ panel 3: heads
x0 = PX[2]
panel(x0, 0, PW[2], H, 'Prediction Heads (every FPN level, shared)', PANEL[2])
cols = [x0 + 25, x0 + 232, x0 + 439]
cw = 172
for c in cols:
    Box(c, 70, cw, 40, "x = Cat(V, A)\n(B, 1024, T')", 'same', size=11)
titles = ['Event Head', 'Boundary Head', 'Embed Head']
kinds = ['new', 'changed', 'new']
for c, t, k in zip(cols, titles, kinds):
    text_lines(c + cw / 2, 135, [t], size=15, weight='bold', color=KIND[k])
towers = []
for c in cols:
    s = stack(c, 160, cw, ['Masked Conv 1D', 'Layer Norm', 'ReLU'], size=12)
    towers.append(s)
    note(c + cw + 4, 230, ['x2'], w=24)
# event
e1 = Box(cols[0], 380, cw, 48, 'Masked Conv 1D -> 1\nbias: prior 0.01', 'new', size=11)
down(towers[0][-1], e1)
e2 = Box(cols[0], 480, cw, 48, "event logit (B, T', 1)\nfocal loss", 'new', size=11)
down(e1, e2)
# boundary
bx = cols[1]
b1 = Box(bx, 380, 80, 48, 'MConv 1D\n-> 2', 'same', size=11)
b2 = Box(bx + 92, 380, 80, 48, 'MConv 1D\n-> 1', 'new', size=11)
down(towers[1][-1], b1); down(towers[1][-1], b2)
b3 = Box(bx, 470, 80, 34, 'Scale_l', 'same', size=11)
b4 = Box(bx, 545, 80, 34, 'ReLU', 'same', size=11)
down(b1, b3); down(b3, b4)
b5 = Box(bx, 620, 80, 48, '(d_s, d_e)\ndistances', 'same', size=11)
down(b4, b5)
b6 = Box(bx + 92, 620, 80, 48, 'IoU quality\nlogit', 'new', size=11)
down(b2, b6)
# embed
mx = cols[2]
m1 = Box(mx, 380, cw, 48, 'Masked Conv 1D\nvis_proj (512)', 'new', size=11)
down(towers[2][-1], m1)
m2 = Box(mx, 480, 80, 48, 'L2 norm', 'new', size=11)
m3 = Box(mx + 92, 480, 80, 48, 'level 0 only:\nraw features', 'new', size=10)
down(m1, m2); down(m1, m3)
m4 = Box(mx, 580, 80, 48, "e_t (B, T', 512)\nper step", 'new', size=10)
down(m2, m4)
m5 = Box(mx + 92, 580, 80, 48, '-> Segment\nContext', 'new', size=11)
down(m3, m5)
m6 = Box(mx, 690, cw, 48, 'clip_proj: Linear 1536 -> 512\n(caption vectors -> same space)', 'new', size=11)
note(mx + cw / 2, 765, ['caption vectors = ONE-PEACE text', 'of each train caption (frozen)'], anchor='middle', w=cw + 20, size=11)
note(x0 + 20, 900, [
    'Event Head thay cho head phân lớp của UniAV gốc:',
    'chỉ 1 kênh "có sự kiện ở bước này không", không theo lớp.',
    '',
    'Boundary Head giữ nhánh hồi quy (d_s, d_e) của UniAV',
    'và thêm nhánh dự đoán IoU giữa đoạn của bước này và GT,',
    'dùng để xếp hạng đoạn khi suy luận.',
    '',
    'Embed Head cho mỗi bước một vector trong không gian',
    'caption; bản ở mức FPN 0 (chưa chuẩn hoá) được',
    'lấy trung bình theo đoạn trong SegmentContext.'], size=13, color=TEXT, w=620)

# ------------------------------------------------------------------ panel 2: segment context + retrieval
x0 = PX[1]
panel(x0, 0, PW[1], H, 'SegmentContext & Caption Retrieval', PANEL[1])
cx = x0 + PW[1] / 2
r0 = Box(cx - 300, 70, 260, 44, "raw level-0 features\n(512, T0)", 'new', size=11)
r1 = Box(cx + 40, 70, 260, 44, 'segment [s, e]\n(grid units)', 'new', size=11)
s1 = Box(cx - 330, 165, 190, 48, 'span mean over [s, e]\n(cumulative sum)', 'new', size=11)
s2 = Box(cx - 95, 165, 190, 48, 'whole-video mean\n(which dish)', 'new', size=11)
s3 = Box(cx + 140, 165, 190, 48, 'position (s/T, e/T)\n(early / late step)', 'new', size=11)
down(r0, s1); down(r0, s2); down(r1, s3); down(r1, s1)
cc = Box(cx - 150, 265, 300, 34, 'Concat -> 2 x 512 + 2', 'new', size=12)
for s_ in (s1, s2, s3):
    down(s_, cc)
mlp = stack(cx - 150, 340, 300, ['Linear 1026 -> 512', 'GELU', 'Linear 512 -> 512 (zero init)'], 'new', h=32, gap=16)
down(cc, mlp[0])
add = Box(cx - 40, 500, 80, 34, '+', 'new', size=16, bold=True)
down(mlp[-1], add)
arrow([(s1.x + 20, s1.y + s1.h), (s1.x + 20, 517), (add.x, 517)], 'residual', lpos=0.0, lside='right')
q = Box(cx - 150, 575, 300, 40, 'L2 norm -> segment vector q (512)', 'new', size=12)
down(add, q)
# retrieval
p1 = Box(cx - 330, 680, 300, 52, 'caption pool: clip_proj(ONE-PEACE text)\n8218 train captions, (N, 512)', 'new', size=11)
p2 = Box(cx + 30, 680, 300, 52, 'omni_proj: Linear 512 -> 1024,\nGELU, Linear 1024 -> 3584, L2', 'new', size=11)
down(q, p1); down(q, p2)
c1 = Box(cx - 330, 790, 300, 40, 's1 = cos(q, caption pool)', 'new', size=12)
c2 = Box(cx + 30, 790, 300, 52, 's2 = cos(q_omni, OmniRetriever-7B\ntext vectors of the same pool)', 'new', size=11)
down(p1, c1); down(p2, c2)
ss = Box(cx - 150, 900, 300, 34, 's = s1 + s2   (N,)', 'new', size=12)
down(c1, ss); down(c2, ss)
tk = Box(cx - 150, 980, 300, 34, 'top 20 candidates', 'new', size=12)
down(ss, tk)
mb = Box(cx - 230, 1060, 460, 52, 'MBR: p = softmax(scale * s_top)\npick argmax_j  sum_k p_k cos(c_j, c_k)', 'new', size=12)
down(tk, mb)
oc = Box(cx - 150, 1160, 300, 34, 'caption of the event', 'same', size=12, bold=True)
down(mb, oc)
note(x0 + 20, 1260, [
    'MLP bắt đầu bằng 0, nên ban đầu q chính là trung bình đoạn.',
    'MBR: chọn câu được nhiều ứng viên mạnh "đồng thuận",',
    'thay vì câu đứng đầu do may mắn.'], size=13, color=TEXT, w=660)

# ------------------------------------------------------------------ panel 1: training losses
x0 = PX[0]
panel(x0, 0, PW[0], H, 'Training objectives', PANEL[0])
cx = x0 + PW[0] / 2
rows = [
    ('L_event', 'sigmoid focal loss: event / no event per step', '1.0'),
    ('L_reg', 'centre-DIoU 1D on (d_s, d_e), positive steps', '1.0'),
    ('L_iou', 'BCE(IoU-quality logit, IoU(pred, GT))', '1.0'),
    ('L_emb', 'per step: CE over the WHOLE caption pool,\nsoft targets', '0.2'),
    ('L_span', 'q of the GT span and of a +-20% jittered span:\nCE over the whole pool, soft targets', '0.2'),
    ('L_omni_txt', 'q_omni vs OmniRetriever text vectors\nof the pool: CE, soft targets', '0.2'),
    ('L_omni_av', 'q_omni vs the OmniRetriever-7B audio+video\nvector of every GT clip: CE (teacher)', '0.2'),
]
y = 75
boxes = []
for name, desc, w in rows:
    n_ = Box(x0 + 25, y, 120, 54, name, 'loss', size=13, bold=True)
    d_ = Box(x0 + 160, y, 400, 54, desc, 'loss', size=11)
    note(x0 + 600, y + 27, ['w = ' + w], w=70, color=TEXT, size=12)
    boxes.append(n_)
    y += 78
tot = Box(x0 + 25, y + 15, 535, 40, 'L = sum of w * L   (AdamW, 2 warm-up + 10 cosine epochs)', 'loss', size=12, bold=True)
st = Box(x0 + 25, y + 95, 535, 66, 'soft target over the pool, for the true caption g:\n'
         't_j = 0.5 * onehot(g)_j + 0.5 * softmax_j( cos(c_g, c_j) / 0.02 )\n(paraphrases of the same step are not punished)', 'loss', size=11)
note(x0 + 20, y + 290, [
    'Không còn loss phân lớp của UniAV gốc.',
    'Hai loss L_omni_* chỉ bật khi có vector thầy',
    '(dataset.omni_emb_file); thầy OmniRetriever-7B đóng băng,',
    'chỉ dùng lúc train, suy luận không cần chạy thầy.',
    '',
    'Kho caption (negatives lúc train và nguồn chọn câu lúc',
    'suy luận) chỉ lấy từ tập train.'], size=13, color=TEXT, w=640)

# ============================================================================== write
mx_xml = ('<mxfile host="tools/draw_architecture.py"><diagram name="UniAV new" id="uniav-new"><mxGraphModel dx="%d" '
          'dy="%d" grid="0" gridSize="10" page="0" background="%s"><root><mxCell id="0"/><mxCell id="1" parent="0"/>%s'
          '</root></mxGraphModel></diagram></mxfile>' % (W, H, BG, ''.join(cells)))
doc = ('<?xml version="1.0" encoding="UTF-8"?>\n'
       '<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="%dpx" height="%dpx" viewBox="0 0 %d %d" '
       'content="%s">\n<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
       'orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#e6e6e6"/></marker></defs>\n'
       '<rect width="100%%" height="100%%" fill="%s"/>\n%s\n</svg>\n'
       % (W, H, W, H, esc(mx_xml), BG, '\n'.join(svg)))
with open(OUT, 'w') as f:
    f.write(doc)
print('wrote %s (%d cells)' % (OUT, len(cells)))
