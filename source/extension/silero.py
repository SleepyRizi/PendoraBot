import os
import torch
import torchaudio
import logging
import re
from num2words import num2words
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

try:
    import extensions.telegram_bot.source.utils as utils
    from extensions.telegram_bot.source.user import User as User
except ImportError:
    import source.utils as utils
    from source.user import User as User

class MySilero:
    punctuation = r"[\s,.?!/)\'\]>]"
    alphabet_map = {
        "A": " Ei ",
        "B": " Bee ",
        "C": " See ",
        "D": " Dee ",
        "E": " Eee ",
        "F": " Eff ",
        "G": " Jee ",
        "H": " Eich ",
        "I": " Eye ",
        "J": " Jay ",
        "K": " Kay ",
        "L": " El ",
        "M": " Emm ",
        "N": " Enn ",
        "O": " Ohh ",
        "P": " Pee ",
        "Q": " Queue ",
        "R": " Are ",
        "S": " Ess ",
        "T": " Tee ",
        "U": " You ",
        "V": " Vee ",
        "W": " Double You ",
        "X": " Ex ",
        "Y": " Why ",
        "Z": " Zed ",
    }
    voices = {
        "en": {
            "model": "v3_en",
            "female": [
                "Kaki",
            ],
        },
        "ru": {
            "model": "v3_1_ru",
            "male": [
                "aidar",
                "eugene",
            ],
            "female": [
                "baya",
                "kseniya",
                "xenia",
            ],
        },
    }

    def __init__(self):
        # Set up XTTS model
        self.base_dir = "/content/llm_telegram_bot"
        self.config_path = os.path.join(self.base_dir, "config.json")
        self.tokenizer_path = os.path.join(self.base_dir, "vocab.json")
        self.xtts_checkpoint = os.path.join(self.base_dir, "model.pth")
        self.speaker_reference = os.path.join(self.base_dir, "kakivoice enhanced_00000001.wav")
        
        # Load XTTS model
        self.config = XttsConfig()
        self.config.load_json(self.config_path)
        self.model = Xtts.init_from_config(self.config)
        self.model.load_checkpoint(self.config, checkpoint_path=self.xtts_checkpoint, vocab_path=self.tokenizer_path, use_deepspeed=False, checkpoint_dir=self.base_dir)
        self.model.cuda()
        
        print("XTTS model loaded successfully.")

    @utils.async_wrap
    def get_audio(self, text: str, user_id: int, user: User):
        try:
            print("Computing speaker latents...")
            gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(audio_path=[self.speaker_reference])

            print("Inference...")
            out = self.model.inference(
                text,
                "en",  # You can modify this part to use dynamic language settings if needed.
                gpt_cond_latent,
                speaker_embedding,
                temperature=0.75  # Add custom parameters here
            )
            wav_path = str(user_id) + ".wav"
            torchaudio.save(wav_path, torch.tensor(out["wav"]).unsqueeze(0), 24000)
            return wav_path
        except Exception as e:
            print(e)
            return None

    def get_default_audio_settings(self, language, sex="female"):
        if language in self.voices:
            return self.voices[language]["model"], self.voices[language][sex][0]
        else:
            return "None", "None"

    def preprocess(self, string):
        # the order for some of these matter
        # For example, you need to remove the commas in numbers before expanding them
        string = self.remove_surrounded_chars(string)
        string = string.replace('"', "")
        string = string.replace("\u201D", "").replace("\u201C", "")  # right and left quote
        string = string.replace("\u201F", "")  # italic looking quote
        string = string.replace("\n", " ")
        string = string.replace("*", " ! ")
        string = self.convert_num_locale(string)
        string = self.replace_negative(string)
        string = self.replace_roman(string)
        string = self.hyphen_range_to(string)
        string = self.num_to_words(string)

        # For now, expand abbreviations to pronunciations
        # replace_abbreviations adds a lot of unnecessary whitespace to ensure separation
        string = self.replace_abbreviations(string)
        string = self.replace_lowercase_abbreviations(string)

        # cleanup whitespaces
        # remove whitespace before punctuation
        string = re.sub(rf"\s+({self.punctuation})", r"\1", string)
        string = string.strip()
        # compact whitespace
        string = " ".join(string.split())

        return string

    @staticmethod
    def remove_surrounded_chars(string):
        # first this expression will check if there is a string nested exclusively between a alt=
        # and a style= string. This would correspond to only a the alt text of an embedded image
        # If it matches it will only keep that part as the string, and rend it for further processing
        # Afterwards this expression matches to 'as few symbols as possible (0 upwards) between any
        # asterisks' OR' as few symbols as possible (0 upwards) between an asterisk and the end of the string'
        if re.search(r"(?<=alt=)(.*)(?=style=)", string, re.DOTALL):
            m = re.search(r"(?<=alt=)(.*)(?=style=)", string, re.DOTALL)
            string = m.group(0)
        return re.sub(r"\*[^*]*?(\*|$)", "", string)

    @staticmethod
    def convert_num_locale(text):
        # This detects locale and converts it to American without comma separators
        pattern = re.compile(r"(?:\s|^)\d{1,3}(?:\.\d{3})+(,\d+)(?:\s|$)")
        result = text
        while True:
            match = pattern.search(result)
            if match is None:
                break

            start = match.start()
            end = match.end()
            result = result[0:start] + result[start:end].replace(".", "").replace(",", ".") + result[end: len(result)]

        # removes comma separators from existing American numbers
        pattern = re.compile(r"(\d),(\d)")
        result = pattern.sub(r"\1\2", result)

        return result

    def replace_negative(self, string):
        # handles situations like -5. -5 would become negative 5, which would then be expanded to negative five
        return re.sub(rf"(\s)(-)(\d+)({self.punctuation})", r"\1negative \3\4", string)

    def replace_roman(self, string):
        # find a string of roman numerals.
        # Only 2 or more, to avoid capturing I and single character abbreviations, like names
        pattern = re.compile(rf"\s[IVXLCDM]{{2,}}{self.punctuation}")
        result = string
        while True:
            match = pattern.search(result)
            if match is None:
                break

            start = match.start()
            end = match.end()
            result = (
                    result[0: start + 1]
                    + str(self.roman_to_int(result[start + 1: end - 1]))
                    + result[end - 1: len(result)]
            )
        return result

    @staticmethod
    def roman_to_int(s):
        rom_val = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
        int_val = 0
        for i in range(len(s)):
            if i > 0 and rom_val[s[i]] > rom_val[s[i - 1]]:
                int_val += rom_val[s[i]] - 2 * rom_val[s[i - 1]]
            else:
                int_val += rom_val[s[i]]
        return int_val

    @staticmethod
    def hyphen_range_to(text):
        pattern = re.compile(r"(\d+)[-–](\d+)")
        result = pattern.sub(lambda x: x.group(1) + " to " + x.group(2), text)
        return result

    @staticmethod
    def num_to_words(text):
        # 1000 or 10.23
        pattern = re.compile(r"\d+\.\d+|\d+")
        result = pattern.sub(lambda x: num2words(float(x.group())), text)
        return result

    def replace_abbreviations(self, string):
        # abbreviations 1-4 characters long. It will get things like A and I, but those are pronounced with their letter
        pattern = re.compile(rf"(^|[\s(.\'\[<])([A-Z]{{1,4}})({self.punctuation}|$)")
        result = string
        while True:
            match = pattern.search(result)
            if match is None:
                break

            start = match.start()
            end = match.end()
            result = result[0:start] + self.replace_abbreviation(result[start:end]) + result[end: len(result)]

        return result

    def replace_lowercase_abbreviations(self, string):
        # abbreviations 1 to 4 characters long, separated by dots i.e. e.g.
        pattern = re.compile(rf"(^|[\s(.\'\[<])(([a-z]\.){{1,4}})({self.punctuation}|$)")
        result = string
        while True:
            match = pattern.search(result)
            if match is None:
                break

            start = match.start()
            end = match.end()
            result = result[0:start] + self.replace_abbreviation(result[start:end].upper()) + result[end: len(result)]

        return result

    def replace_abbreviation(self, string):
        result = ""
        for char in string:
            result += self.match_mapping(char)

        return result

    def match_mapping(self, char):
        for mapping in self.alphabet_map.keys():
            if char == mapping:
                return self.alphabet_map[char]

        return char

    def __main__(self, args):
        print(self.preprocess(args[1]))
