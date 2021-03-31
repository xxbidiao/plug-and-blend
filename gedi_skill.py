"""
gedi_skill.py

This file contains wrapper for our blending generative model, improved upon GeDi.

Some code is based upon https://colab.research.google.com/github/salesforce/GeDi/blob/master/GeDi_guided_GPT_2_XL.ipynb

"""

import numpy as np
import torch
from nltk import sent_tokenize
from gedi_helpers.modeling_gpt2 import GPT2LMHeadModel
from transformers import (
    GPT2Config,
    GPT2Tokenizer
)

# pad token id used to signify this token should be masked.
pad_token_for_masking = 0


def cut_into_sentences(text, do_cleanup=True):
    """
    Cut text into sentences. \n are also regarded as a sentence.
    :param do_cleanup: if True, do cleanups.
    :param text: input text.
    :return: sentences.
    """
    all_sentences = []
    # sentences_raw = text.split("\n")
    sentences_raw = [text.replace("\n", " ")]
    result = []

    for item in sentences_raw:
        sentence_in_item = sent_tokenize(item)
        for item2 in sentence_in_item:
            all_sentences.append(item2)

    if do_cleanup:
        for item in all_sentences:
            item = item.replace('<|endoftext|>', '')
            if len(item) > 2:
                result.append(item)
    else:
        result = all_sentences
    return result


class GediSkill:
    __instance = None

    @staticmethod
    def getUniqueInstance(*args, **kwargs):
        """ Static access method. Calling this twice will get the old instance. """
        if GediSkill.__instance == None:
            GediSkill.__instance = GediSkill(*args, **kwargs)
        return GediSkill.__instance

    def __init__(self, custom_model_path=None,gedi_model_path = None):
        """
        Initialize this gedi skill.
        :param custom_model_path: path to base model.
        """
        self.mode = "topic"
        self.code_desired = "true"
        self.code_undesired = "false"
        self.model_type = 'gpt2'
        self.gen_type = "gedi"
        self.gen_model_name_or_path = "gpt2-large"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.MODEL_CLASSES = {"gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer), }
        self.config_class, self.model_class, self.tokenizer_class = self.MODEL_CLASSES["gpt2"]
        self.tokenizer = self.tokenizer_class.from_pretrained(self.gen_model_name_or_path, do_lower_case=False)

        if custom_model_path is None:
            self.model = self.model_class.from_pretrained(self.gen_model_name_or_path, load_in_half_prec=True)
        else:
            self.model = self.model_class.from_pretrained(custom_model_path)
        self.model = self.model.to(self.device)
        self.model = self.model.float()

        gedi_model_name_or_path = gedi_model_path
        self.gedi_model = self.model_class.from_pretrained(gedi_model_name_or_path)
        self.gedi_model.to(self.device)

        # setting arguments for generation
        # max generation length, in tokens.
        self.gen_length = 50
        # omega from original GeDi work, higher disc_weight means more aggressive topic steering.
        # can be overridden when calling generate_one_sentence(), see that function.
        # default value (1x) is 30.
        self.disc_weight = 30
        # 1 - rho from GeDi. Not used, kept for compatibility.
        self.filter_p = 0.8
        # tau from original GeDi. Not used.
        self.target_p = 0.8
        # hyperparameter that determines class prior, set to uniform by default
        self.class_bias = 0

        if self.gen_length > 1024:
            self.length = 1024
        else:
            self.length = self.gen_length

    def __call__(self, data):
        """
        Compatible interface for Ray library.
        :param data: data: dict that contains "sentence' and "topic" key.
        :return: Generated text.
        """
        return self.generate_one_sentence_skill(data)

    def prepare_tokens(self, sentences):
        """
        Get tokenized representation of a sentence.
        :param sentences: input sentence.
        :return: tokenized representation.
        """
        self.tokenizer.pad_token = 50256
        self.tokenizer.padding_side = "left"
        return self.tokenizer(sentences, padding=True)

    def get_base_model_prob(self, sentence):
        """
        Get the probability (or a form of perplexity) of `sentence` seen by the base LM.
        :param sentence: input sentence.
        :return: probability.
        """
        text_ids = self.tokenizer.encode(sentence)
        text_ids_tensor = torch.LongTensor(text_ids).unsqueeze(0).to(self.device)
        output = self.model(text_ids_tensor, labels=text_ids_tensor)
        return float(np.exp(np.average(output[0].cpu().detach().numpy())))  # loss

    def generate_one_sentence_skill(self, data):
        """
        Generate one sentence based on input data.
        :param data: dict that contains "sentence' and "topic" key.
        :return: Generated text, in Ray-API compatible format.
        """
        # Repack the data
        result = self.generate_one_sentence(data['sentence'], data['topic'])
        return {
            "in_sentence": data['sentence'],
            "in_topic": data["topic"],
            "out_sentence": result,
        }

    def generate_one_sentence(self, sentence, topic, extra_args=None):
        """
        Generate one sentence based on input data.
        :param sentence: (string) context (prompt) used.
        :param topic: (dict) {topic: weight, topic:weight,...} topic that the sentence need to steer towards.
        :param extra_args: (dict) a dictionary that certain key will trigger additional functionality.
            disc_weight: Set this value to use a different control strength than default.
            get_gen_token_count: Return only how many tokens the generator has generated (for debug only).
        :return: sentence generated, or others if extra_args are specified.
        """
        secondary_code = topic

        disc_weight = self.disc_weight
        if type(extra_args) is dict and 'disc_weight' in extra_args:
            disc_weight = extra_args['disc_weight']

        if sentence == "":
            print("Prompt is empty! Using a dummy sentence.")
            sentence = "."

        # Specify prompt below
        prompt = sentence

        # Calculate oroginal input length.
        length_of_prompt = len(sentence)

        start_len = 0
        text_ids = self.tokenizer.encode(prompt)
        length_of_prompt_in_tokens = len(text_ids)
        encoded_prompts = torch.LongTensor(text_ids).unsqueeze(0).to(self.device)

        if type(topic) is str:
            multi_code = self.tokenizer.encode(secondary_code)
        elif type(topic) is dict:
            multi_code = {}
            for item in secondary_code:
                encoded = self.tokenizer.encode(item)[0]  # only take the first one
                multi_code[encoded] = secondary_code[item]
        else:
            raise NotImplementedError("topic data type of %s not supported... Supported: (str,dict)" % type(topic))

        # If 1, generate sentences towards a specific topic.
        self.attr_class = 1

        generated_sequence = self.model.generate(input_ids=encoded_prompts,
                                                 pad_lens=None,
                                                 max_length=self.length + length_of_prompt_in_tokens,
                                                 top_k=None,
                                                 top_p=None,
                                                 repetition_penalty=1.2,
                                                 rep_penalty_scale=10,
                                                 eos_token_ids=self.tokenizer.eos_token_id,
                                                 pad_token_id=0,  # self.tokenizer.eos_token_id,
                                                 do_sample=False,
                                                 penalize_cond=True,
                                                 gedi_model=self.gedi_model,
                                                 tokenizer=self.tokenizer,
                                                 disc_weight=disc_weight,
                                                 filter_p=self.filter_p,
                                                 target_p=self.target_p,
                                                 class_bias=self.class_bias,
                                                 attr_class=self.attr_class,
                                                 code_0=self.code_undesired,
                                                 code_1=self.code_desired,
                                                 multi_code=multi_code,
                                                 )

        if type(extra_args) is dict and 'get_gen_token_count' in extra_args:
            return len(generated_sequence.tolist()[0])

        text = self.tokenizer.decode(generated_sequence.tolist()[0], clean_up_tokenization_spaces=True,
                                     skip_special_tokens=True)

        text = text[length_of_prompt:]
        text = cut_into_sentences(text)
        if len(text) == 0:
            print("Warning! No text generated.")
            return ""
        all_gen_text = text[0]
        return all_gen_text

    def generate_sentences(self, sentences, topic):
        """
        DOESN'T WORK - Do not use this function.
        Currently only correctly supports sentences that tokenizes into the same number of tokens.
        Doesn't work with default GeDi padding scheme. Needs a GeDi model that is trained by right-padding GeDi-prompts.
        TODO: fixing it by changing how positional embedding works for GeDi model.
        Only supports batch generation with same number of tokens for default GeDi model.
        :param sentences:
        :param topic:
        :return:
        """
        secondary_code = topic

        for index, sentence in enumerate(sentences):
            if sentence == "":
                print("One of the prompt is empty! Using a dummy sentence.")
                sentences[index] = "."

        # # Specify prompt below
        # prompt = sentences

        # text_ids = self.tokenizer.encode(prompt)
        # length_of_prompt_in_tokens = len(text_ids)
        # encoded_prompts = torch.LongTensor(text_ids).unsqueeze(0).to(self.device)

        prepared_tokens_dict = self.prepare_tokens(
            sentences=sentences,
        )

        input_ids = torch.tensor(prepared_tokens_dict['input_ids'])
        masks = torch.tensor(prepared_tokens_dict['attention_mask'])

        encoded_prompts = input_ids.to(self.device)

        if type(topic) is str:
            multi_code = self.tokenizer.encode(secondary_code)
        elif type(topic) is dict:
            multi_code = {}
            for item in secondary_code:
                encoded = self.tokenizer.encode(item)[0]  # only take the first one
                multi_code[encoded] = secondary_code[item]
        else:
            raise NotImplementedError("topic data type of %s not supported... Supported: (str,dict)" % type(topic))

        # If 1, generate sentences towards a specific topic.
        self.attr_class = 1

        # custom_eos_token_id = self.tokenizer.encode(".")[0]
        # print("Using custom eos_token_id: %s" % custom_eos_token_id)

        # print("LENGTH_OF_PROMPT = %s"%length_of_prompt_in_tokens)
        # print("multi_code = %s"%str(multi_code))

        generated_sequence = self.model.generate(input_ids=encoded_prompts,
                                                 attention_mask=masks,
                                                 pad_lens=None,
                                                 max_length=self.length * 2,
                                                 top_k=None,
                                                 top_p=None,
                                                 repetition_penalty=1.2,
                                                 rep_penalty_scale=10,
                                                 eos_token_ids=self.tokenizer.eos_token_id,
                                                 pad_token_id=self.tokenizer.eos_token_id,  # 0,
                                                 do_sample=False,
                                                 penalize_cond=True,
                                                 gedi_model=self.gedi_model,
                                                 tokenizer=self.tokenizer,
                                                 disc_weight=self.disc_weight,
                                                 filter_p=self.filter_p,
                                                 target_p=self.target_p,
                                                 class_bias=self.class_bias,
                                                 attr_class=self.attr_class,
                                                 code_0=self.code_undesired,
                                                 code_1=self.code_desired,
                                                 multi_code=multi_code,
                                                 )

        text = []
        for item in generated_sequence:
            # print("raw generated: %s"%str(item))
            text.append(
                self.tokenizer.decode(item.tolist(), clean_up_tokenization_spaces=True, skip_special_tokens=True))
        result = []
        print(text)
        for index, input_sentence in enumerate(sentences):
            this_text = text[index][len(input_sentence):]
            # print("Prompt = %s, topic = %s, raw text = %s" % (prompt[-10:], topic, text))
            this_text = cut_into_sentences(this_text)
            if len(this_text) == 0:
                print("No text generated.")
                result.append("")
            result.append(this_text[0])
        return result

    def generate_one_probability_skill(self, data):
        """
                Gives probability for a sentence seen by the GeDi model (for debug).
        :param data: dict that contains "sentence' and "topic" key.
        :return: Generated text, in Ray-API compatible format.
        """
        # print("DATA = %s"%data)
        # Repack the data
        result = self.generate_one_probability(data['this_sentence'], data['topic'])
        return {
            "in_sentence": data['sentence'],
            "in_topic": data["topic"],
            "out_probs": result,
        }

    def generate_one_probability(self, sentence, topic):
        """
        Gives probability for a sentence seen by the GeDi model (for debug).
        :param sentence:
        :param topic:
        :return:
        """
        # print("Input: s=%s t=%s"%(sentence,topic))
        secondary_code = topic

        if type(topic) is not dict:
            raise NotImplementedError(
                "topic parameter is type %s which is not supported for this operation. Supported: (dict)" % type(topic))

        if sentence == "":
            print("Prompt is empty! Set all scores to 0.")
            result = {}
            for key in topic:
                result[key] = 0.0
            return result

        # Specify prompt below
        prompt = sentence

        # Calculate oroginal input length.
        length_of_prompt = len(sentence)

        start_len = 0
        text_ids = self.tokenizer.encode(prompt)
        length_of_prompt_in_tokens = len(text_ids)
        encoded_prompts = torch.LongTensor(text_ids).unsqueeze(0).to(self.device)

        encoded_to_original = {}
        if type(topic) is str:
            multi_code = self.tokenizer.encode(secondary_code)
        elif type(topic) is dict:
            multi_code = {}
            for item in secondary_code:
                encoded = self.tokenizer.encode(item)[0]  # only take the first one
                multi_code[encoded] = secondary_code[item]
                encoded_to_original[encoded] = item

        self.attr_class = 1

        probs = self.model.generate(input_ids=encoded_prompts,
                                    pad_lens=None,
                                    max_length=self.length + length_of_prompt_in_tokens,
                                    top_k=None,
                                    top_p=None,
                                    repetition_penalty=1.2,
                                    rep_penalty_scale=10,
                                    eos_token_ids=self.tokenizer.eos_token_id,
                                    pad_token_id=0,
                                    do_sample=False,
                                    penalize_cond=True,
                                    gedi_model=self.gedi_model,
                                    tokenizer=self.tokenizer,
                                    disc_weight=self.disc_weight,
                                    filter_p=self.filter_p,
                                    target_p=self.target_p,
                                    class_bias=self.class_bias,
                                    attr_class=self.attr_class,
                                    code_0=self.code_undesired,
                                    code_1=self.code_desired,
                                    multi_code=multi_code,
                                    only_get_gedi_probs=True, # This additional parameter trigger different function
                                    )
        # normalization - make the highest score 1 for ease of use
        max_ = max(probs.values())
        if max_ < 1e-9:
            max_ = 1e-9
        results = {}
        for key in probs:
            if key not in encoded_to_original:
                raise RuntimeError("Can't find which topic a prob score is for from API.")
            results[encoded_to_original[key]] = probs[key] / max_
        return results
