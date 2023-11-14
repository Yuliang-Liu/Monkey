# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tokenization classes for QWen."""

import base64
import logging
import os
import requests
import unicodedata
from typing import Collection, Dict, List, Set, Tuple, Union, Any, Callable, Optional

import tiktoken
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from transformers import PreTrainedTokenizer, AddedToken
from transformers.utils import try_to_load_from_cache

import matplotlib.colors as mcolors
from matplotlib.font_manager import FontProperties

logger = logging.getLogger(__name__)


VOCAB_FILES_NAMES = {"vocab_file": "qwen.tiktoken", "ttf": "SimSun.ttf"}

PAT_STR = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
ENDOFTEXT = "<|endoftext|>"
IMSTART = "<|im_start|>"
IMEND = "<|im_end|>"
# as the default behavior is changed to allow special tokens in
# regular texts, the surface forms of special tokens need to be
# as different as possible to minimize the impact
EXTRAS = tuple((f"<|extra_{i}|>" for i in range(205)))
SPECIAL_TOKENS = (
    ENDOFTEXT,
    IMSTART,
    IMEND,
) + EXTRAS
IMG_TOKEN_SPAN = 1280


def _load_tiktoken_bpe(tiktoken_bpe_file: str) -> Dict[bytes, int]:
    with open(tiktoken_bpe_file, "rb") as f:
        contents = f.read()
    return {
        base64.b64decode(token): int(rank)
        for token, rank in (line.split() for line in contents.splitlines() if line)
    }

def _list_find(
    input_list: List[Any],
    candidates: Tuple[Any],
    start: int = 0,
):
    for i in range(start, len(input_list)):
        if input_list[i] in candidates:
            return i
    return -1

def _replace_closed_tag(
    input_tokens: List[Any],
    start_tags: Union[Any, Tuple[Any]],
    end_tags: Union[Any, Tuple[Any]],
    inclusive_replace_func: Callable,
    exclusive_replace_func: Callable = lambda x: x,
):
    if isinstance(start_tags, (str, int)):
        start_tags = (start_tags,)
    if isinstance(end_tags, (str, int)):
        end_tags = (end_tags,)
    assert len(start_tags) == len(end_tags)

    output_tokens = []
    end = 0
    while True:
        start = _list_find(input_tokens, start_tags, end)
        if start == -1:
            break
        output_tokens.extend(exclusive_replace_func(input_tokens[end : start]))
        tag_idx = start_tags.index(input_tokens[start])
        end = _list_find(input_tokens, (end_tags[tag_idx],), start)
        if end == -1:
            raise ValueError("Unclosed image token")
        output_tokens.extend(inclusive_replace_func(input_tokens[start : end + 1]))
        end += 1
    output_tokens.extend(exclusive_replace_func(input_tokens[end : ]))
    return output_tokens

class QWenTokenizer(PreTrainedTokenizer):
    """QWen tokenizer."""

    vocab_files_names = VOCAB_FILES_NAMES

    def __init__(
        self,
        vocab_file,
        errors="replace",
        image_start_tag='<img>',
        image_end_tag='</img>',
        image_pad_tag='<imgpad>',
        ref_start_tag='<ref>',
        ref_end_tag='</ref>',
        box_start_tag='<box>',
        box_end_tag='</box>',
        quad_start_tag='<quad>',
        quad_end_tag='</quad>',
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_start_tag = image_start_tag
        self.image_end_tag = image_end_tag
        self.image_pad_tag = image_pad_tag
        self.ref_start_tag = ref_start_tag
        self.ref_end_tag = ref_end_tag
        self.box_start_tag = box_start_tag
        self.box_end_tag = box_end_tag
        self.quad_start_tag = quad_start_tag
        self.quad_end_tag = quad_end_tag
        self.IMAGE_ST = (
            ref_start_tag, ref_end_tag,
            box_start_tag, box_end_tag,
            quad_start_tag, quad_end_tag,
            image_start_tag, image_end_tag,
            image_pad_tag
        )

        self.errors = errors  # how to handle errors in decoding

        self.mergeable_ranks = _load_tiktoken_bpe(vocab_file)  # type: dict[bytes, int]
        self.special_tokens = {
            token: index
            for index, token in enumerate(
                SPECIAL_TOKENS + self.IMAGE_ST, start=len(self.mergeable_ranks)
            )
        }
        self.img_start_id = self.special_tokens[self.image_start_tag]
        self.img_end_id = self.special_tokens[self.image_end_tag]
        self.img_pad_id = self.special_tokens[self.image_pad_tag]
        self.ref_start_id = self.special_tokens[self.ref_start_tag]
        self.ref_end_id = self.special_tokens[self.ref_end_tag]
        self.box_start_id = self.special_tokens[self.box_start_tag]
        self.box_end_id = self.special_tokens[self.box_end_tag]
        self.quad_start_id = self.special_tokens[self.quad_start_tag]
        self.quad_end_id = self.special_tokens[self.quad_end_tag]

        enc = tiktoken.Encoding(
            "Qwen",
            pat_str=PAT_STR,
            mergeable_ranks=self.mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        assert (
            len(self.mergeable_ranks) + len(self.special_tokens) == enc.n_vocab
        ), f"{len(self.mergeable_ranks) + len(self.special_tokens)} != {enc.n_vocab} in encoding"

        self.decoder = {
            v: k for k, v in self.mergeable_ranks.items()
        }  # type: dict[int, bytes|str]
        self.decoder.update({v: k for k, v in self.special_tokens.items()})

        self.tokenizer = enc  # type: tiktoken.Encoding

        self.eod_id = self.tokenizer.eot_token
        self.im_start_id = self.special_tokens[IMSTART]
        self.im_end_id = self.special_tokens[IMEND]

    def __getstate__(self):
        # for pickle lovers
        state = self.__dict__.copy()
        del state['tokenizer']
        return state

    def __setstate__(self, state):
        # tokenizer is not python native; don't pass it; rebuild it
        self.__dict__.update(state)
        enc = tiktoken.Encoding(
            "Qwen",
            pat_str=PAT_STR,
            mergeable_ranks=self.mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        self.tokenizer = enc


    def __len__(self) -> int:
        return self.tokenizer.n_vocab

    def get_vocab(self) -> Dict[bytes, int]:
        return self.mergeable_ranks

    def convert_tokens_to_ids(
        self, tokens: Union[bytes, str, List[Union[bytes, str]]]
    ) -> List[int]:
        ids = []
        if isinstance(tokens, (str, bytes)):
            if tokens in self.special_tokens:
                return self.special_tokens[tokens]
            else:
                return self.mergeable_ranks.get(tokens)
        for token in tokens:
            if token in self.special_tokens:
                ids.append(self.special_tokens[token])
            else:
                ids.append(self.mergeable_ranks.get(token))
        return ids

    def _add_tokens(self, new_tokens: Union[List[str], List[AddedToken]], special_tokens: bool = False) -> int:
        if not special_tokens and new_tokens:
            raise ValueError('Adding regular tokens is not supported')
        for token in new_tokens:
            surface_form = token.content if isinstance(token, AddedToken) else token
            if surface_form not in SPECIAL_TOKENS + self.IMAGE_ST:
                raise ValueError('Adding unknown special tokens is not supported')
        return 0

    def save_vocabulary(self, save_directory: str, **kwargs) -> Tuple[str]:
        """
        Save only the vocabulary of the tokenizer (vocabulary).

        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        file_path = os.path.join(save_directory, "qwen.tiktoken")
        with open(file_path, "w", encoding="utf8") as w:
            for k, v in self.mergeable_ranks.items():
                line = base64.b64encode(k).decode("utf8") + " " + str(v) + "\n"
                w.write(line)
        return (file_path,)

    def tokenize(
        self,
        text: str,
        allowed_special: Union[Set, str] = "all",
        disallowed_special: Union[Collection, str] = (),
        **kwargs,
    ) -> List[Union[bytes, str]]:
        """
        Converts a string in a sequence of tokens.

        Args:
            text (`str`):
                The sequence to be encoded.
            allowed_special (`Literal["all"]` or `set`):
                The surface forms of the tokens to be encoded as special tokens in regular texts.
                Default to "all".
            disallowed_special (`Literal["all"]` or `Collection`):
                The surface forms of the tokens that should not be in regular texts and trigger errors.
                Default to an empty tuple.

            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific encode method.

        Returns:
            `List[bytes|str]`: The list of tokens.
        """
        tokens = []
        text = unicodedata.normalize("NFC", text)

        # this implementation takes a detour: text -> token id -> token surface forms
        for t in self.tokenizer.encode(
            text, allowed_special=allowed_special, disallowed_special=disallowed_special
        ):
            tokens.append(self.decoder[t])

        def _encode_imgurl(img_tokens):
            assert img_tokens[0] == self.image_start_tag and img_tokens[-1] == self.image_end_tag
            img_tokens = img_tokens[1:-1]
            img_url = b''.join(img_tokens)
            out_img_tokens = list(map(self.decoder.get, img_url))
            if len(out_img_tokens) > IMG_TOKEN_SPAN:
                raise ValueError("The content in {}..{} is too long".format(
                    self.image_start_tag, self.image_end_tag))
            out_img_tokens.extend([self.image_pad_tag] * (IMG_TOKEN_SPAN - len(out_img_tokens)))
            out_img_tokens = [self.image_start_tag] + out_img_tokens + [self.image_end_tag]
            return out_img_tokens

        return _replace_closed_tag(tokens, self.image_start_tag, self.image_end_tag, _encode_imgurl)

    def convert_tokens_to_string(self, tokens: List[Union[bytes, str]]) -> str:
        """
        Converts a sequence of tokens in a single string.
        """
        text = ""
        temp = b""
        for t in tokens:
            if isinstance(t, str):
                if temp:
                    text += temp.decode("utf-8", errors=self.errors)
                    temp = b""
                text += t
            elif isinstance(t, bytes):
                temp += t
            else:
                raise TypeError("token should only be of type types or str")
        if temp:
            text += temp.decode("utf-8", errors=self.errors)
        return text

    @property
    def vocab_size(self):
        return self.tokenizer.n_vocab

    def _convert_id_to_token(self, index: int) -> Union[bytes, str]:
        """Converts an id to a token, special tokens included"""
        if index in self.decoder:
            return self.decoder[index]
        raise ValueError("unknown ids")

    def _convert_token_to_id(self, token: Union[bytes, str]) -> int:
        """Converts a token to an id using the vocab, special tokens included"""
        if token in self.special_tokens:
            return self.special_tokens[token]
        if token in self.mergeable_ranks:
            return self.mergeable_ranks[token]
        raise ValueError("unknown token")

    def _tokenize(self, text: str, **kwargs):
        """
        Converts a string in a sequence of tokens (string), using the tokenizer. Split in words for word-based
        vocabulary or sub-words for sub-word-based vocabularies (BPE/SentencePieces/WordPieces).

        Do NOT take care of added tokens.
        """
        raise NotImplementedError

    def _decode(
        self,
        token_ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
        errors: str = None,
        **kwargs,
    ) -> str:
        if isinstance(token_ids, int):
            token_ids = [token_ids]

        def _decode_imgurl(img_token_ids):
            assert img_token_ids[0] == self.img_start_id and img_token_ids[-1] == self.img_end_id
            img_token_ids = img_token_ids[1:-1]
            img_token_ids = img_token_ids[ : img_token_ids.index(self.img_pad_id)]
            img_url = bytes(img_token_ids).decode('utf-8')
            return [self.img_start_id] + self.tokenizer.encode(img_url) + [self.img_end_id]

        token_ids = _replace_closed_tag(token_ids, self.img_start_id, self.img_end_id, _decode_imgurl)

        if skip_special_tokens:
            token_ids = [i for i in token_ids if i < self.eod_id]
        return self.tokenizer.decode(token_ids, errors=errors or self.errors)

    def to_list_format(self, text: str):
        text = unicodedata.normalize("NFC", text)
        token_ids = self.tokenizer.encode(
            text, allowed_special=set(self.IMAGE_ST + (ENDOFTEXT,)))

        def _encode_vl_info(tokens):
            if len(tokens) == 0:
                return []
            if tokens[0] == self.img_start_id and tokens[-1] == self.img_end_id:
                key = 'image'
            elif tokens[0] == self.ref_start_id and tokens[-1] == self.ref_end_id:
                key = 'ref'
            elif tokens[0] == self.box_start_id and tokens[-1] == self.box_end_id:
                key = 'box'
            elif tokens[0] == self.quad_start_id and tokens[-1] == self.quad_end_id:
                key = 'quad'
            else:
                _tobytes = lambda x: x.encode('utf-8') if isinstance(x, str) else x
                return [{'text': b''.join(map(_tobytes, map(self.decoder.get, tokens))).decode('utf-8')}]
            _tobytes = lambda x: x.encode('utf-8') if isinstance(x, str) else x
            val = b''.join(map(_tobytes, map(self.decoder.get, tokens[1:-1]))).decode('utf-8')
            return [{key: val}]

        return _replace_closed_tag(
            token_ids,
            (self.img_start_id, self.ref_start_id, self.box_start_id, self.quad_start_id),
            (self.img_end_id, self.ref_end_id, self.box_end_id, self.quad_end_id),
            _encode_vl_info,
            _encode_vl_info,
        )

    def from_list_format(self, list_format: List[Dict]):
        text = ''
        num_images = 0
        for ele in list_format:
            if 'image' in ele:
                num_images += 1
                text += f'Picture {num_images}:'
                text += self.image_start_tag + ele['image'] + self.image_end_tag
                text += '\n'
            elif 'text' in ele:
                text += ele['text']
            elif 'box' in ele:
                if 'ref' in ele:
                    text += self.ref_start_tag + ele['ref'] + self.ref_end_tag
                for box in ele['box']:
                    text += self.box_start_tag + '(%d,%d),(%d,%d)' % (box[0], box[1], box[2], box[3]) + self.box_end_tag
            else:
                raise ValueError("Unsupport element: " + str(ele))
        return text

    def _fetch_latest_picture(self, response, history):
        if history is None:
            history = []
        _history = history + [(response, None)]
        for q, r in _history[::-1]:
            for ele in self.to_list_format(q)[::-1]:
                if 'image' in ele:
                    return ele['image']
        return None

    def _fetch_all_box_with_ref(self, text):
        list_format = self.to_list_format(text)
        output = []
        for i, ele in enumerate(list_format):
            if 'box' in ele:
                bbox = tuple(map(int, ele['box'].replace('(', '').replace(')', '').split(',')))
                assert len(bbox) == 4
                output.append({'box': bbox})
                if i > 0 and 'ref' in list_format[i-1]:
                    output[-1]['ref'] = list_format[i-1]['ref'].strip()
        return output

    def draw_bbox_on_latest_picture(
        self,
        response,
        history=None,
    ) -> Optional[Image.Image]:
        image = self._fetch_latest_picture(response, history)
        if image is None:
            return None
        if image.startswith("http://") or image.startswith("https://"):
            image = Image.open(requests.get(image, stream=True).raw).convert("RGB")
            h, w = image.height, image.width
        else:
            image = np.asarray(Image.open(image).convert("RGB"))
            h, w = image.shape[0], image.shape[1]
        visualizer = Visualizer(image)

        boxes = self._fetch_all_box_with_ref(response)
        if not boxes:
            return None
        color = random.choice([_ for _ in mcolors.TABLEAU_COLORS.keys()]) # init color
        for box in boxes:
            if 'ref' in box: # random new color for new refexps
                color = random.choice([_ for _ in mcolors.TABLEAU_COLORS.keys()])
            x1, y1, x2, y2 = box['box']
            x1, y1, x2, y2 = (int(x1 / 1000 * w), int(y1 / 1000 * h), int(x2 / 1000 * w), int(y2 / 1000 * h))
            visualizer.draw_box((x1, y1, x2, y2), alpha=1, edge_color=color)
            if 'ref' in box:
                visualizer.draw_text(box['ref'], (x1, y1), color=color, horizontal_alignment="left")
        return visualizer.output


import colorsys
import logging
import math
import numpy as np
import matplotlib as mpl
import matplotlib.colors as mplc
import matplotlib.figure as mplfigure
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
import random

logger = logging.getLogger(__name__)


class VisImage:
    def __init__(self, img, scale=1.0):
        self.img = img
        self.scale = scale
        self.width, self.height = img.shape[1], img.shape[0]
        self._setup_figure(img)

    def _setup_figure(self, img):
        fig = mplfigure.Figure(frameon=False)
        self.dpi = fig.get_dpi()
        # add a small 1e-2 to avoid precision lost due to matplotlib's truncation
        # (https://github.com/matplotlib/matplotlib/issues/15363)
        fig.set_size_inches(
            (self.width * self.scale + 1e-2) / self.dpi,
            (self.height * self.scale + 1e-2) / self.dpi,
        )
        self.canvas = FigureCanvasAgg(fig)
        # self.canvas = mpl.backends.backend_cairo.FigureCanvasCairo(fig)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.axis("off")
        self.fig = fig
        self.ax = ax
        self.reset_image(img)

    def reset_image(self, img):
        img = img.astype("uint8")
        self.ax.imshow(img, extent=(0, self.width, self.height, 0), interpolation="nearest")

    def save(self, filepath):
        self.fig.savefig(filepath)

    def get_image(self):
        canvas = self.canvas
        s, (width, height) = canvas.print_to_buffer()

        buffer = np.frombuffer(s, dtype="uint8")

        img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)
        return rgb.astype("uint8")


class Visualizer:
    def __init__(self, img_rgb, metadata=None, scale=1.0):
        self.img = np.asarray(img_rgb).clip(0, 255).astype(np.uint8)
        self.font_path = try_to_load_from_cache("Qwen/Qwen-VL-Chat", "SimSun.ttf")
        self.output = VisImage(self.img, scale=scale)
        self.cpu_device = torch.device("cpu")

        # too small texts are useless, therefore clamp to 14
        self._default_font_size = max(
            np.sqrt(self.output.height * self.output.width) // 30, 15 // scale
        )

    def draw_text(
        self,
        text,
        position,
        *,
        font_size=None,
        color="g",
        horizontal_alignment="center",
        rotation=0,
    ):
        if not font_size:
            font_size = self._default_font_size

        # since the text background is dark, we don't want the text to be dark
        color = np.maximum(list(mplc.to_rgb(color)), 0.2)
        color[np.argmax(color)] = max(0.8, np.max(color))

        x, y = position
        self.output.ax.text(
            x,
            y,
            text,
            size=font_size * self.output.scale,
            fontproperties=FontProperties(fname=self.font_path),
            bbox={"facecolor": "black", "alpha": 0.8, "pad": 0.7, "edgecolor": "none"},
            verticalalignment="top",
            horizontalalignment=horizontal_alignment,
            color=color,
            zorder=10,
            rotation=rotation,
        )
        return self.output

    def draw_box(self, box_coord, alpha=0.5, edge_color="g", line_style="-"):
        
        x0, y0, x1, y1 = box_coord
        width = x1 - x0
        height = y1 - y0

        linewidth = max(self._default_font_size / 4, 1)

        self.output.ax.add_patch(
            mpl.patches.Rectangle(
                (x0, y0),
                width,
                height,
                fill=False,
                edgecolor=edge_color,
                linewidth=linewidth * self.output.scale,
                alpha=alpha,
                linestyle=line_style,
            )
        )
        return self.output

    def get_output(self):
        
        return self.output
