import datetime
from tqdm import tqdm
from typing import List

from sarathi import LLMEngine, SamplingParams, RequestOutput

BASE_OUTPUT_DIR = "./offline_inference_output"

# Sample prompts.
# prompts = [
#     "Hello, my name is",
#     "The president of the United States is",
#     "The capital of France is",
#     "The future of AI is",
# ]
prompts = [
    "The immediate reaction in some circles of the archeological community was that the accuracy of our dating was insufficient to make the extraordinary claim that humans were present in North America during the Last Glacial Maximum. But our targeted methodology in this current research really paid off, said Jeff Pigati, USGS research geologist and co-lead author of a newly published study that confirms the age of the White Sands footprints. The controversy centered on the accuracy of the original ages, which were obtained by radiocarbon dating. The age of the White Sands footprints was initially determined by dating seeds of the common aquatic plant Ruppia cirrhosa that were found in the fossilized impressions. But aquatic plants can acquire carbon from dissolved carbon atoms in the water rather than ambient air, which can potentially cause the measured ages to be too old. Even as the original work was being published, we were forging ahead to test our results with multiple lines of evidence, said Kathleen Springer, USGS research geologist and co-lead author on the current Science paper. We were confident in our original ages, as well as the strong geologic, hydrologic, and stratigraphic evidence, but we knew that independent chronologic control was critical.",
    "The breakthrough technique developed by University of Oxford researchers could one day provide tailored repairs for those who suffer brain injuries. The researchers demonstrated for the first time that neural cells can be 3D printed to mimic the architecture of the cerebral cortex. The results have been published today in the journal Nature Communications. Brain injuries, including those caused by trauma, stroke and surgery for brain tumours, typically result in significant damage to the cerebral cortex (the outer layer of the human brain), leading to difficulties in cognition, movement and communication. For example, each year, around 70 million people globally suffer from traumatic brain injury (TBI), with 5 million of these cases being severe or fatal. Currently, there are no effective treatments for severe brain injuries, leading to serious impacts on quality of life. Tissue regenerative therapies, especially those in which patients are given implants derived from their own stem cells, could be a promising route to treat brain injuries in the future. Up to now, however, there has been no method to ensure that implanted stem cells mimic the architecture of the brain.",
    "Hydrogen ions are the key component of acids, and as foodies everywhere know, the tongue senses acid as sour. That's why lemonade (rich in citric and ascorbic acids), vinegar (acetic acid) and other acidic foods impart a zing of tartness when they hit the tongue. Hydrogen ions from these acidic substances move into taste receptor cells through the OTOP1 channel. Because ammonium chloride can affect the concentration of acid -- that is, hydrogen ions -- within a cell, the team wondered if it could somehow trigger OTOP1. To answer this question, they introduced the Otop1 gene into lab-grown human cells so the cells produce the OTOP1 receptor protein. They then exposed the cells to acid or to ammonium chloride and measured the responses. We saw that ammonium chloride is a really strong activator of the OTOP1 channel, Liman said. It activates as well or better than acids. Ammonium chloride gives off small amounts of ammonia, which moves inside the cell and raises the pH, making it more alkaline, which means fewer hydrogen ions.",
    "Ava is planning a camping trip with her friends. She wants to make sure they have enough granola bars for snacks. There will be five people total: Ava, her two friends, and her parents. They will spend 3 days and 2 nights at the campsite, and they plan to have 2 granola bars per person for breakfast and 1 granola bar per person for an afternoon snack each day. How many granola bars will Ava need to pack in total for the entire trip?",
    "You will be given a definition of a task first, then some input of the task. This task is about using the specified sentence and converting the sentence to Resource Description Framework (RDF) triplets of the form (subject, predicate object). The RDF triplets generated must be such that the triplets accurately capture the structure and semantics of the input sentence. The input is a sentence and the output is a list of triplets of the form [subject, predicate, object] that capture the relationships present in the sentence. When a sentence has more than 1 RDF triplet possible, the output must contain all of them. AFC Ajax (amateurs)'s ground is Sportpark De Toekomst where Ajax Youth Academy also play. Output:",
    "What happens next in this paragraph? She then rubs a needle on a cotton ball then pushing it onto a pencil and wrapping thread around it. She then holds up a box of a product and then pouring several liquids into a bowl. she Choose your answer from: A. adds saucepan and shakes up the product in a grinder. B. pinches the thread to style a cigarette, and then walks away. C. then dips the needle in ink and using the pencil to draw a design on her leg, rubbing it off with a rag in the end. D. begins to style her hair and cuts it several times before parting the ends of it to show the hairstyle she has created.",
    "Write a question about the following article: Coming off their home win over the Buccaneers, the Packers flew to Ford Field for a Week 12 Thanksgiving duel with their NFC North foe, the Detroit Lions. After a scoreless first quarter, Green Bay delivered the game's opening punch in the second quarter with quarterback Aaron Rodgers finding wide receiver Greg Jennings on a 3-yard touchdown pass. The Packers added to their lead in the third quarter with a 1-yard touchdown run from fullback John Kuhn, followed by Rodgers connecting with wide receiver James Jones on a 65-yard touchdown pass and a 35-yard field goal from kicker Mason Crosby. The Lions answered in the fourth quarter with a 16-yard touchdown run by running back Keiland Williams and a two-point conversion pass from quarterback Matthew Stafford to wide receiver Titus Young), yet Green Bay pulled away with Crosby nailing a 32-yard field goal. Detroit closed out the game with Stafford completing a 3-yard touchdown pass to wide receiver Calvin Johnson. With the win, the Packers acquired their first 11-0 start in franchise history, beating the 1962 team which started 10-0 and finished 14-1 including postseason play. Rodgers (22/32 for 307 yards, 2 TDs) was named NFL on FOX's 2011 Galloping Gobbler Award Winner. Question about the article:",
    "Luckily I was not the target or victim of this road rage, but I was a first hand witness. Back in the mid 1990s when I worked in Southern California, I was returning from lunch one day with a coworker when we noticed two pickup trucks driving erratically. This was on Valley View Blvd in Cerritos which is a large roadway divided (by a large, planted median strip) with 3 lanes on either side. One truck seemed to be chasing the other when suddenly the lead truck made a quick U-turn and was followed by the second truck. But the second truck came to stop just after making the turn. This was when the driver of that truck got out, walked to the front of his vehicle, raised his arm holding a semi-automatic handgun, paused, and then fired a single shot. At this point, I was parallel with the shooter’s truck and only about 50–60 feet from where he was standing. I remember my coworker exclaiming, “Oh sh*t! Did he just shoot at the something?!” I floored my accelerator and sped off to my workplace which was about a 1/2 mile down Valley View Blvd. I parked and my coworker and I got into the building as quickly as we could. While he started telling our other coworkers about what we had just witnessed, I ran to my desk, called 911, and reported the shooting. The dispatcher kept me on the line after my initial report and then transferred me to a detective in order to take down my information. As it turns out, my information and the fact that I called 911 immediately resulted in the arrest of the shooter. I was later asked to testify in court which resulted in the shooters conviction. I lived in Southern California for 10 years and now I drive a commercial truck several days a week so I have PLENTY of other “road rage stories”, but this is still the craziest one. According to the above context, choose the correct option to answer the following question. Question: How long did it take the driver to get to his workplace? Options: - one minute - not enough information - one year - one day",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)

output_dir = f"{BASE_OUTPUT_DIR}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

llm_engine = LLMEngine.from_engine_args(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    # parallel config
    tensor_parallel_size=1,
    pipeline_parallel_size=2,
    trust_remote_code=True,
    max_model_len=4096,
    # scheduler config
    scheduler_type="sarathi",
    chunk_size=100,
    max_num_seqs=4,
    # metrics config
    write_metrics=True,
    output_dir=output_dir,
    enable_chrome_trace=True,
    enable_profiler=True
)


def generate(
    llm_engine: LLMEngine,
    prompts: List[str],
    sampling_params: SamplingParams,
) -> List[RequestOutput]:
    for prompt in prompts:
        llm_engine.add_request(prompt, sampling_params)

    num_requests = llm_engine.get_num_unfinished_requests()
    pbar = tqdm(total=num_requests, desc="Processed prompts")
    # Run the engine
    outputs: List[RequestOutput] = []
    while llm_engine.has_unfinished_requests():
        step_outputs = llm_engine.step()
        for output in step_outputs:
            if output.finished:
                outputs.append(output)
                pbar.update(1)
    pbar.close()
    # Sort the outputs by request ID.
    # This is necessary because some requests may be finished earlier than
    # its previous requests.
    outputs = sorted(outputs, key=lambda x: int(x.seq_id))
    return outputs


# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = generate(llm_engine, prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.text
    print("===========================================================")
    print(f"Prompt: {prompt!r}")
    print("-----------------------------------------------------------")
    print(f"Generated text: {generated_text!r}")
    print("===========================================================")
llm_engine.exit()
