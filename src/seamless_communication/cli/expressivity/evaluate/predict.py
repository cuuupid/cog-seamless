import contextlib
import logging
from argparse import Namespace
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torchaudio
from fairseq2.data import Collater, DataPipeline, FileMapper
from fairseq2.data.audio import (
    AudioDecoder,
    WaveformToFbankConverter,
    WaveformToFbankOutput,
)
from fairseq2.data.text import StrSplitter, read_text
from fairseq2.generation import NGramRepeatBlockProcessor
from fairseq2.typing import DataType, Device
from torch import Tensor
from tqdm import tqdm

from seamless_communication.cli.expressivity.evaluate.pretssel_inference_helper import (
    PretsselGenerator,
)
from seamless_communication.cli.m4t.evaluate.evaluate import (
    adjust_output_for_corrupted_inputs,
    count_lines,
)

from seamless_communication.inference import (
    SequenceGeneratorOptions,
    BatchedSpeechOutput,
    Translator,
)
from seamless_communication.models.unity import (
    load_gcmvn_stats,
    load_unity_unit_tokenizer,
)
from seamless_communication.store import add_gated_assets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

def go(_src: str, src_lang: str, tgt_lang: str):
    src = Path(_src)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        dtype = torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    unit_tokenizer = load_unity_unit_tokenizer("seamless_expressivity")
    _gcmvn_mean, _gcmvn_std = load_gcmvn_stats("vocoder_pretssel")
    gcmvn_mean = torch.tensor(_gcmvn_mean, device=device, dtype=dtype)
    gcmvn_std = torch.tensor(_gcmvn_std, device=device, dtype=dtype)

    add_gated_assets(Path("./model/"))

    mapper = FileMapper(root_dir=".", cached_fd_count=10)
    decoder = AudioDecoder(dtype=torch.float32, device=device)
    wav_to_fbank = WaveformToFbankConverter(
        num_mel_bins=80,
        waveform_scale=2**15,
        channel_last=True,
        standardize=False,
        device=device,
        dtype=dtype,
    )
    def normalize_fbank(data: WaveformToFbankOutput) -> WaveformToFbankOutput:
        fbank = data["fbank"]
        std, mean = torch.std_mean(fbank, dim=0)
        data["fbank"] = fbank.subtract(mean).divide(std)
        data["gcmvn_fbank"] = fbank.subtract(gcmvn_mean).divide(gcmvn_std)
        return data
    collater = Collater(pad_value=0, pad_to_multiple=1)
    translator = Translator(
        "seamless_expressivity",
        vocoder_name_or_card=None,
        device=device,
    )

    fbank = collater(normalize_fbank(wav_to_fbank(decoder(mapper(src)["data"]))))

    text_generation_opts = SequenceGeneratorOptions(
        beam_size=5,
        soft_max_seq_len=(1,200),
        unk_penalty=torch.inf
    )
    unit_generation_opts = SequenceGeneratorOptions(
        beam_size=5,
        soft_max_seq_len=(25,50),
    )

    pretssel_generator = PretsselGenerator(
        "vocoder_pretssel",
        vocab_info=unit_tokenizer.vocab_info,
        device=device,
        dtype=dtype,
    )

    output_path = Path('./tmp') / src.stem
    output_path.mkdir(parents=True, exist_ok=True)

    waveforms_dir = output_path / "waveform"
    waveforms_dir.mkdir(parents=True, exist_ok=True)

    hyps = []
    refs = []
    audio_hyps = []

    with contextlib.ExitStack() as stack:
        hyp_file = stack.enter_context(
            open(output_path / f"text_output-{src.stem}.txt", "w")
        )
        unit_file = stack.enter_context(
            open(output_path / f"unit_output-{src.stem}.txt", "w")
        )

        valid_sequences: Optional[Tensor] = None
        source = fbank["fbank"]

        # Skip corrupted audio tensors.
        valid_sequences = ~torch.any(
            torch.any(torch.isnan(source["seqs"]), dim=1), dim=1
        )

        if not valid_sequences.all():
            logger.warning(
                f"Sample has some corrupted input."
            )
            source["seqs"] = source["seqs"][valid_sequences]
            source["seq_lens"] = source["seq_lens"][valid_sequences]

        # Skip performing inference when the input is entirely corrupted.
        if source["seqs"].numel() > 0:
            prosody_encoder_input = fbank["gcmvn_fbank"]
            text_output, unit_output = translator.predict(
                source,
                "s2st",
                tgt_lang,
                src_lang=src_lang,
                text_generation_opts=text_generation_opts,
                unit_generation_opts=unit_generation_opts,
                unit_generation_ngram_filtering=False,
                duration_factor=1.0,
                prosody_encoder_input=prosody_encoder_input,
            )

            assert unit_output is not None
            speech_output = pretssel_generator.predict(
                unit_output.units,
                tgt_lang=tgt_lang,
                prosody_encoder_input=prosody_encoder_input,
            )

        else:
            text_output = []
            speech_output = BatchedSpeechOutput(units=[], audio_wavs=[])

        if valid_sequences is not None and not valid_sequences.all():
            text_output, speech_output = adjust_output_for_corrupted_inputs(  # type: ignore[assignment]
                valid_sequences,
                text_output,
                speech_output,
            )

        hyps += [str(s) for s in text_output]

        for i in range(len(text_output)):
            t = text_output[i]
            hyp_file.write(f"{t}\n")

            u = speech_output.units[i]
            str_units = [str(i) for i in u]
            unit_file.write(" ".join(str_units) + "\n")
            torchaudio.save(
                waveforms_dir / f"pred.wav",
                speech_output.audio_wavs[i][0].to(torch.float32).cpu(),
                sample_rate=speech_output.sample_rate,
            )
            audio_hyps.append((waveforms_dir / f"pred.wav").as_posix())



