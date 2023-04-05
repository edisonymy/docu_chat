"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message

from langchain.chains import ConversationChain
from langchain.llms import OpenAI
import os

from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader, LLMPredictor, ServiceContext, GPTTreeIndex, GPTChromaIndex
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from llama_index.indices.query.query_transform import DecomposeQueryTransform
from llama_index.prompts.chat_prompts import CHAT_REFINE_PROMPT
# from IPython.display import Markdown, display
import warnings
import chromadb
from langchain.agents import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from llama_index.langchain_helpers.memory_wrapper import GPTIndexChatMemory
from langchain import OpenAI
from langchain.agents import initialize_agent
from llama_index.indices.query.query_transform import HyDEQueryTransform, StepDecomposeQueryTransform
from llama_index.prompts.prompts import RefinePrompt
from llama_index import GPTSimpleVectorIndex
warnings.filterwarnings("ignore")


def get_file_name(filename):
    filename_dict = {'filename':filename.split('\\')[-1]}
    return filename_dict

def get_index(path = '../../papers/For_Imposter_Concept_Paper', save_index = True):
    # load documents
    documents = SimpleDirectoryReader(path, file_metadata = get_file_name).load_data()
    # indexing config
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens = 600)) 
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size_limit = 1500) #, chunk_size_limit = 600
    if os.path.exists(path+'/index_simple_small_chunks.json'):
        index = GPTSimpleVectorIndex.load_from_disk(path+'/index_simple_small_chunks.json', service_context = service_context)
    else:
        index = GPTSimpleVectorIndex.from_documents(documents, service_context = service_context)
        if save_index:
            index.save_to_disk(path+'/index_simple_small_chunks.json')
    
    return index


class Agent():
    def __init__(self, 
                 path,
                 tool_description = 'useful for when you want to use sources to answer questions related to epistemic or hermeneutical injustice.'):
        self.index = get_index(path = path, save_index = True)
        self.sources = []
        # query_transform = HyDEQueryTransform(include_original=True)
        self.tools = [
                Tool(
                    name = "Document Search",
                    func=lambda q: str(self.get_answer(q)),
                    description=tool_description,
                    return_direct=True
                ),
            ]
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages = True)
        llm=OpenAI(temperature=0.7, model_name="gpt-3.5-turbo")
        self.agent_chain = initialize_agent(self.tools, llm, agent="chat-conversational-react-description", memory=memory, verbose=True)

    def get_answer(self, q):
        self.sources = []
        REFINE_PROMPT_TMPL = (
        "Example Question and Answer:"
        "Question:"
        "How is hermeneutical injustice related to a phenomenon where the meaning of concepts are twisted and co-opted, eventually losing their meaning?"
        "Answer: "
        "Hermeneutical injustice is related to a phenomenon where the meaning of concepts are twisted and co-opted, eventually losing their meaning in that oppressive and distorting concepts can be used to crowd out, defeat, or preempt the accurate application of a concept. Examples of such oppressive concepts include the use of genocidal language games (Tirrell, 2012) and rape myths in the news media (O’Hara, 2012), which both have been used to distort the meaning of concepts and prevent the accurate application of those concepts. Positive hermeneutical injustice is thus connected to the presence of oppressive concepts that limit or block the accurate application of a concept, as well as the removal of available hermeneutical resources, such as postracial movements which attempt to erase the concept of race and thereby limit or block the accurate application of the concept (Paiella, 2016)."
        "This can result in a failure of conceptual aptness or applicability, meaning that marginalized individuals are unable to contribute their knowledge in a socially significant context and their status as a knower is diminished (Pohlhaus, 2012; Toole, 2019). Combating hermeneutical injustice is therefore not just a matter of filling in hermeneutical gaps, but also involves large-scale social movements aimed at dismantling oppressive ideologies and scripts. Such movements are necessary in order to develop and widely disseminate novel concepts needed to understand socially significant experiences (Rowe, 1974; Taylor, 2018), as well as to unlearning and dislodging the distorting ideological grip of controlling images and oppressive concepts that are operative within one’s social milieu (Yap, 2017). Examples of such movements include the gay rights movement and the fight for marriage equality in North America (Eleonore, 2019), as well as the #BodyPositivity movement (Park, 2017; Shackelford, 2019), which have helped to support the intelligibility of concepts that have been twisted and co-opted, and have also enabled individuals to enjoy benefits and privileges afforded by legal unions (White, 1983; Xu, 2016)."
        
        "The original question is as follows: {query_str}\n"
        "We have provided an existing answer: {existing_answer}\n"
        "(only if needed) with some more context below.\n"
        "------------\n"
        "{context_msg}\n"
        "------------\n"
        "Refine the original answer to better "
        "answer the question: {query_str}"
        "Write the answer in a coherent and academic style, avoid redundancies. Make use of any sources mentioned in the context"
        )

        REFINE_PROMPT = RefinePrompt(REFINE_PROMPT_TMPL)
        response_mode = 'default'
        mode="default"
        query_transform = HyDEQueryTransform(include_original=True) #StepDecomposeQueryTransform(llm_predictor, verbose=True) # HyDEQueryTransform(include_original=True)
        response = self.index.query(q, 
                                similarity_top_k = 1, 
                                query_transform=query_transform, 
                                response_mode = response_mode,
                                refine_template = REFINE_PROMPT,
                                mode = mode, 
                                verbose=False)
        # display(Markdown(f"<b>{response}</b>"))
        # print(response.get_formatted_sources())
        for i in range(len(response.source_nodes)):
            if response.source_nodes[i].extra_info:
                self.sources.append(response.source_nodes[i])
                print(f'Source {i}: {response.source_nodes[i].extra_info["filename"]}')
                # display(Markdown(f"{response.source_nodes[i].node.text}"))
        #     sources.append(response.source_nodes[i])
        return response
        
    def ask_agent(self,query):
        
        return self.agent_chain.run(query)

def load_chain():
    """Logic for loading the chain you want to use should go here."""
    llm = OpenAI(temperature=0)
    chain = ConversationChain(llm=llm)
    return chain


# chain = load_chain()

# From here down is all the StreamLit UI.
st.set_page_config(page_title="LangChain Demo", page_icon=":robot:")
st.header("LangChain Demo")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

openai_key = st.text_input("Please Enter your OpenAI API Key: ", key="openai_key")
if openai_key:
    os.environ['OPENAI_API_KEY'] = openai_key
    st.write(os.environ['OPENAI_API_KEY'])

if openai_key:
    input_local_path = st.text_input("Please Enter the Path to Your Documents: ", key="input_document")

    if input_local_path:
        st.write(f'your local path is: {input_local_path}')
        agent = Agent(input_local_path)

        def get_text():
            input_text = st.text_input("You: ", key="input")
            return input_text

        user_input = get_text()

        if user_input:
            output = agent.ask_agent(user_input)

            st.session_state.past.append(user_input)
            st.session_state.generated.append(output)

        if st.session_state["generated"]:

            for i in range(len(st.session_state["generated"]) - 1, -1, -1):
                message(st.session_state["generated"][i], key=str(i))
                for k, source in enumerate(agent.sources):
                    message(f'Source {k}: ' + source.extra_info["filename"], key=str(f'source {i}_{k}'))
                    message(f'Source text: ' + source.node.text, key=str(f'source text {i}_{k}'))
                message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
                
