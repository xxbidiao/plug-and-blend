import numpy as np
from scipy.stats import norm
# Specify which LM interface is used.
from gedi_skill import GediSkill

class Feedback:
    def __init__(self,opcode,content):
        self.opcode = opcode
        self.content = content

# region Direct Agent

# Maximum summary types to support, by default.
default_summary_types = 4

directagent_apply_summary = 0
directagent_request_summary = 1000
directagent_generate = 1
directagent_end_interaction = -1

# This was only for debugging. Set these locations by yourself, or use the demo entry point.
try:
    from local_config import gedi_location,base_LM_path
except ImportError:
    print("Local location variables not used.")

class GeDiStoryAgent():
    """
    This agent assumes summary is 1:1 mapped to skills.
    It directly decides which skill to use by the input summary, with gaussian prior assumption.
    """

    # A sentence as context when there are not,
    # or when the generator can't generate anything (due to a natural finish in input hinting topic changes).
    dummy_sentence = "Recently, "

    def __init__(
            self,
            base_model_path,
            gedi_model_path,
            gen_length=10,
            gaussian_var=2,
            weight_mode=None,
            summary_type_names=None,
            generation_extra_horizon=1,
    ):
        """
        Create an agent interface for generating stories.
        :param base_model_path: (String) location to base LM.
        :param gen_length: (int) Maximum tokens to be generated (passed to generator).
        :param gaussian_var: (float) \sigma in the paper.
        :param weight_mode: None or "linear" (experimental), change how weight is passed to the generator.
        :param summary_type_names: List[String], Supported summary types.
        :param generation_extra_horizon: (int) sentences to append as extra context in iterative generation process (default 1, append up to 1 more previously generated sentences)
        """
        self.gedi_skill_obj = GediSkill.getUniqueInstance(
            custom_model_path=base_model_path,
            gedi_model_path=gedi_model_path,
        )

        self.generation_extra_horizon = generation_extra_horizon
        self.gen_length = gen_length
        self.summary_types_count = len(summary_type_names)
        self.summary = np.zeros((gen_length, self.summary_types_count))
        self.next_action = directagent_request_summary
        self.gaussian_var = gaussian_var
        self.last_feedback = None
        self.summary_type_names = summary_type_names
        self.summary_type_dict = {}
        self.weight_mode = weight_mode
        for item in self.summary_type_names:
            self.summary_type_dict[item] = 1.0  # dummy values

        # Place to put generated sentences to work in feedback loop.
        self.generated_sentences = []

        # Whether in editing process a sentence is forced by the user.
        self.no_regenration_mask = [False] * gen_length

    def receive_feedback(self, feedback_object: Feedback):
        epsilon = 1e-7
        if feedback_object.opcode == directagent_apply_summary:
            content = feedback_object.content
            if content['summary_type'] >= 0:
                type_of_summary = content['summary_type']
                start = content['start']
                end = content['end'] + epsilon  # so that we support single sentence "areas"
                center = (start + end) / 2
                total_summary_values = 0
                for i in range(self.gen_length):
                    # Convert absolute position to where we look into in PDF
                    relative_pdf_pos = 1.0 * (i - center) / (end - center)  # end = 1, start = -1
                    pdf_position = self.gaussian_var * relative_pdf_pos
                    pdf_value = norm.pdf(pdf_position)
                    self.summary[i, type_of_summary] += pdf_value
                    total_summary_values += self.summary[i, type_of_summary]
                for i in range(self.gen_length):
                    # Normalize it if multiple summary of the same type is provided.
                    self.summary[i, type_of_summary] /= total_summary_values
                self.next_action = directagent_request_summary
            else:
                self.next_action = directagent_generate
        else:
            raise RuntimeError

    def do_next_action(self):
        if self.next_action == directagent_request_summary:
            self.request_info(Feedback(directagent_request_summary, None))
        elif self.next_action == directagent_apply_summary:
            self.receive_feedback(self.last_feedback)
        elif self.next_action == directagent_generate:
            self.generate()
        elif self.next_action == directagent_end_interaction:
            print("Done, thank you for using!")
            return False
        return True

    def request_info(self, info_object):
        if info_object.opcode == directagent_request_summary:
            summary_type_input = input("Starting a new sketch. Input index of topic, or no input if no more new sketches:")
            if summary_type_input == "":
                summary_type_input = -1
            try:
                summary_type = int(summary_type_input)
                if self.summary_type_names is not None and summary_type in range(len(self.summary_type_names)):
                    print("This sketch is for topic: %s" % self.summary_type_names[summary_type])
            except ValueError:
                raise
            if summary_type < 0:  # no more summary
                self.next_action = directagent_generate
                return
            start = int(input("Area to apply, start?"))
            end = int(input("Area to apply, end?"))
            feedback = Feedback(directagent_apply_summary,
                                {
                                    'summary_type': summary_type,
                                    'start': start,
                                    'end': end,
                                })
            self.next_action = directagent_apply_summary
            self.last_feedback = feedback
        else:
            raise RuntimeError

    def generate(self, mode="blend"):
        print("Now generating...")
        # print(self.summary)
        if mode == "naive":
            skills_to_use = []
            skills_to_use_raw = np.argmax(self.summary, axis=1)
            for item in skills_to_use_raw:
                skills_to_use.append({self.summary_type_names[item]: 1})
        elif mode == "blend":
            skills_to_use = []
            for item in self.summary:  # for each sentence
                normalized_item = item / np.sum(item)
                for idx, value in enumerate(normalized_item):
                    if value < 0.1 / len(self.summary_type_names):
                        normalized_item[idx] = 0
                # normalize it again
                normalized_item = normalized_item / np.sum(normalized_item)
                temp_item = {}
                for subitem_index in range(len(normalized_item)):  # for each topic
                    if normalized_item[subitem_index] > 0.001:  # epsilon
                        value = normalized_item[subitem_index]
                        if self.weight_mode is None:
                            pass  # value is good already
                        elif self.weight_mode == "linear":
                            value = np.log10(value)
                            # since value is now (-3,0]
                            value = value + 3
                        temp_item[self.summary_type_names[subitem_index]] = value
                skills_to_use.append(temp_item)
        else:
            raise NotImplementedError("generation mode %s not supported: %s" % mode)
        print("Planner output: %s" % skills_to_use)
        last_sentence = self.dummy_sentence
        all_sentences = []
        import tqdm
        index = -1
        for idx in tqdm.tqdm(skills_to_use):
            index += 1
            context = last_sentence
            if self.generation_extra_horizon > 0:
                skip_first = True  # to skip the prompt already attached
                sentence_attached = 0
                for item in reversed(all_sentences):
                    if skip_first:
                        skip_first = False
                        continue
                    sentence_attached += 1
                    context = item + " " + context
                    if sentence_attached >= self.generation_extra_horizon:
                        #print("Attached %s additional sentence as context." % sentence_attached)
                        break
            if not self.no_regenration_mask[index]:
                sentence = self.gedi_skill_obj.generate_one_sentence(sentence=context,topic=idx)
                if len(sentence) < 1:
                    # No sentence generated, assuming force of change of topics
                    sentence = self.gedi_skill_obj.generate_one_sentence(sentence=self.dummy_sentence, topic=idx)
                all_sentences.append(sentence)
            else:
                output = self.generated_sentences[index]
                sentence = output
                all_sentences.append(sentence)
            last_sentence = sentence

        # write all sentences to working sentence "memory"
        self.generated_sentences = all_sentences
        for idx in range(len(skills_to_use)):
            if self.summary_type_names is not None and skills_to_use[idx] in range(len(self.summary_type_names)):
                summary_name = self.summary_type_names[skills_to_use[idx]]
            else:
                summary_name = "Configuration %s" % skills_to_use[idx]
            print("[Sentence %s is using %s]\n%s" % (idx, summary_name, all_sentences[idx]))

        print("-----")
        for item in all_sentences:
            print(item)
        print("-----")
        self.next_action = directagent_end_interaction


# endregion

def run_demo(topics = None,base_language_model_path = None, gedi_path = None):
    """
    Run demo.
    :param topics: Topics to choose from. Can be a list of any strings (suggested to be 1-token words).
    :return: None.
    """
    if topics is None:
        topics_available = ["Sports", "Science"]
    else:
        topics_available = topics

    if base_language_model_path is None:
        try:
            base_language_model_path = base_LM_path
        except:
            print("Info - Now using default (gpt-2-large) as base model.")


    if gedi_path is None:
        try:
            gedi_path = gedi_location
        except:
            raise RuntimeError("GeDi model Path missing.")

    print("Topics available: %s (Configure it in code)" % topics_available)
    agent = GeDiStoryAgent(
        base_model_path=base_language_model_path,
        gedi_model_path=gedi_path,
        summary_type_names=topics_available,
        gen_length=10,
        gaussian_var=1,
        generation_extra_horizon=1,
        weight_mode=None,
       )
    while agent.do_next_action():
        pass

if __name__ == '__main__':
    run_demo()

