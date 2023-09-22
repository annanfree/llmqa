import shutil
from chains.local_doc_qa import LocalDocQA
from configs.model_config import *
import nltk
import models.shared as shared
from models.loader.args import parser
from models.loader import LoaderCheckPoint
import os
import threading
import json
import gradio as gr
from starlette.requests import Request
from fastapi import FastAPI, Form
from starlette.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.responses import RedirectResponse
import json

nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path
final_history = ""
# last_query = ""
USER_INFO_PATH = "/home/bnrcsvr/qy/project2/glmlc/user_info"
# 创建一个全局锁，用于控制文件访问
file_lock = threading.Lock()


def login_func(username, password):
    if username == password:
        return True
    else:
        return False


def get_vs_list():
    lst_default = ["新建知识库"]
    if not os.path.exists(KB_ROOT_PATH):
        return lst_default
    lst = os.listdir(KB_ROOT_PATH)
    if not lst:
        return lst_default
    lst.sort()
    return lst_default + lst


embedding_model_dict_list = list(embedding_model_dict.keys())

llm_model_dict_list = list(llm_model_dict.keys())

local_doc_qa = LocalDocQA()
print(f"local_doc_qa.llm_model_chain={local_doc_qa.llm_model_chain}")

flag_csv_logger = gr.CSVLogger()


def _answer_record(username, qa, chatbot_info, cur_chat_id):
    print(f"username = {username}")
    print(f"qa = {qa}")
    print(f"chatbot_info = {chatbot_info}")
    print(f"cur_chat_id = {cur_chat_id}")
    cur_chat_index = 0
    # 更新chatbot_info
    for cur_chat_index, chat in enumerate(chatbot_info["info"]):
        if chat[0] == cur_chat_id:
            chatbot_info["info"][cur_chat_index][1].append(qa)
            break

    # 会话名设置为第一个问题,同时修改会话记录
    # 如果修改的会话名是当前会话，记录新的会话名
    chat_list = [x[0] for x in chatbot_info["info"]]
    chat_title = chatbot_info["info"][cur_chat_index][1][0][0] \
        if chatbot_info["info"][cur_chat_index][1][0][0] \
        else chatbot_info["info"][cur_chat_index][1][1][0]
    chat_title = process_title(chat_title)
    if chat_list[cur_chat_index] != chat_title:
        chat_list[cur_chat_index] = chat_title
        chatbot_info["info"][cur_chat_index][0] = chat_title
        # cur_chat_id = chat_title

    # 记录新会话
    record_user_info(username, cur_id=cur_chat_id, chat_id=chat_title, new_chat=qa)

    return chat_list, chat_title, chatbot_info


def get_answer(req: gr.Request, query, vs_path, history, mode, admin_flag, chatbot_info, cur_chat_id,
               score_threshold=VECTOR_SEARCH_SCORE_THRESHOLD, vector_search_top_k=VECTOR_SEARCH_TOP_K,
               chunk_conent: bool = True, chunk_size=CHUNK_SIZE, streaming: bool = STREAMING):
    if mode == "Bing搜索问答":
        for resp, history in local_doc_qa.get_search_result_based_answer(
                query=query, chat_history=history, streaming=streaming):
            source = "\n\n"
            source += "".join(
                [
                    f"""<details> <summary>出处 [{i + 1}] <a href="{doc.metadata["source"]}" target="_blank">{doc.metadata["source"]}</a> </summary>\n"""
                    f"""{doc.page_content}\n"""
                    f"""</details>"""
                    for i, doc in
                    enumerate(resp["source_documents"])])
            history[-1][-1] += source
            yield history, "", gr.update(visible=False), gr.update(visible=False)
    elif mode == "知识库问答" and vs_path is not None and os.path.exists(vs_path) and "index.faiss" in os.listdir(
            vs_path):
        global final_history
        exception_flag = False
        try:
            generator = local_doc_qa.get_knowledge_based_answer(query=query, vs_path=vs_path, chat_history=history,
                                                                streaming=streaming)
            while True:
                # generator = local_doc_qa.get_knowledge_based_answer(query=query, vs_path=vs_path, chat_history=history, streaming=streaming)
                resp, history = next(generator)
                if admin_flag:
                    source = "\n\n"
                    source += "".join(
                        [
                            f"""<details> <summary>出处 [{i + 1}] {os.path.split(doc.metadata["source"])[-1]}</summary>\n"""
                            f"""{doc.page_content}\n"""
                            f"""</details>"""
                            for i, doc in
                            enumerate(resp["source_documents"])])
                    history[-1][-1] += source
                # print(f"history:{history}")
                final_history = history
                yield history, "", gr.update(visible=True), gr.update(visible=False), \
                      gr.update(), chatbot_info
        except Exception as e:
            print(e)
        except GeneratorExit:
            print("GeneratorExit")
            exception_flag = True
            chat_list, chat_title, chatbot_info = _answer_record(req.username, final_history[-1], chatbot_info, cur_chat_id)
            return final_history, "", gr.update(visible=False), gr.update(visible=True), \
                    gr.update(choices=chat_list, value=chat_title), chatbot_info
        finally:
            print("finally")
            if not exception_flag:
                chat_list, chat_title, chatbot_info = _answer_record(req.username, final_history[-1], chatbot_info, cur_chat_id)
                return final_history, "", gr.update(visible=False), gr.update(visible=True), \
                    gr.update(choices=chat_list, value=chat_title), chatbot_info

    elif mode == "知识库测试":
        if os.path.exists(vs_path):
            resp, prompt = local_doc_qa.get_knowledge_based_conent_test(query=query, vs_path=vs_path,
                                                                        score_threshold=score_threshold,
                                                                        vector_search_top_k=vector_search_top_k,
                                                                        chunk_conent=chunk_conent,
                                                                        chunk_size=chunk_size)
            if not resp["source_documents"]:
                yield history + [[query,
                                  "根据您的设定，没有匹配到任何内容，请确认您设置的知识相关度 Score 阈值是否过小或其他参数是否正确。"]], "", gr.update(
                    visible=False), gr.update(visible=False)
            else:
                source = "\n".join(
                    [
                        f"""<details open> <summary>【知识相关度 Score】：{doc.metadata["score"]} - 【出处{i + 1}】：  {os.path.split(doc.metadata["source"])[-1]} </summary>\n"""
                        f"""{doc.page_content}\n"""
                        f"""</details>"""
                        for i, doc in
                        enumerate(resp["source_documents"])])
                history.append([query, "以下内容为知识库中满足设置条件的匹配结果：\n\n" + source])
                yield history, "", gr.update(visible=False), gr.update(visible=False)
        else:
            yield history + [[query,
                              "请选择知识库后进行测试，当前未选择知识库。"]], "", gr.update(visible=False), gr.update(
                visible=False)
    else:

        answer_result_stream_result = local_doc_qa.llm_model_chain(
            {"prompt": query, "history": history, "streaming": streaming})

        for answer_result in answer_result_stream_result['answer_result_stream']:
            resp = answer_result.llm_output["answer"]
            history = answer_result.history
            history[-1][-1] = resp
            yield history, "", gr.update(visible=False), gr.update(visible=False)
    logger.info(f"flagging: username={FLAG_USER_NAME},query={query},vs_path={vs_path},mode={mode},history={history}")
    if not isinstance(mode, str):
        flag_csv_logger.flag([query, vs_path, history, mode], username=FLAG_USER_NAME)


# def get_answer_user(query, vs_path, history, streaming: bool = STREAMING):
#     if vs_path is not None and os.path.exists(vs_path) and "index.faiss" in os.listdir(
#             vs_path):
#         for resp, history in local_doc_qa.get_knowledge_based_answer(
#                 query=query, vs_path=vs_path, chat_history=history, streaming=streaming):
#             source = "\n\n"
#             source += "".join(
#                 [f"""<details> <summary>出处 [{i + 1}] {os.path.split(doc.metadata["source"])[-1]}</summary>\n"""
#                  f"""{doc.page_content}\n"""
#                  f"""</details>"""
#                  for i, doc in
#                  enumerate(resp["source_documents"])])
#             history[-1][-1] += source
#             yield history, ""


def init_model():
    args = parser.parse_args()
    args_dict = vars(args)
    shared.loaderCheckPoint = LoaderCheckPoint(args_dict)
    llm_model_ins = shared.loaderLLM()
    try:
        local_doc_qa.init_cfg(llm_model=llm_model_ins)
        answer_result_stream_result = local_doc_qa.llm_model_chain(
            {"prompt": "你好", "history": [], "streaming": False})

        for answer_result in answer_result_stream_result['answer_result_stream']:
            print(f"answer_result.llm_output: {answer_result.llm_output}\n")
        reply = """模型已成功加载，可以开始对话"""
        logger.info(reply)
        return reply
    except Exception as e:
        logger.error(e)
        reply = """模型未成功加载，请到页面左上角"模型配置"选项卡中重新选择后点击"加载模型"按钮"""
        if str(e) == "Unknown platform: darwin":
            logger.info("该报错可能因为您使用的是 macOS 操作系统，需先下载模型至本地后执行 Web UI，具体方法请参考项目 README 中本地部署方法及常见问题："
                        " https://github.com/imClumsyPanda/langchain-ChatGLM")
        else:
            logger.info(reply)
        return reply


def reinit_model(llm_model, embedding_model, llm_history_len, no_remote_model, use_ptuning_v2, use_lora, top_k,
                 history):
    try:
        llm_model_ins = shared.loaderLLM(llm_model, no_remote_model, use_ptuning_v2)
        llm_model_ins.history_len = llm_history_len
        local_doc_qa.init_cfg(llm_model=llm_model_ins,
                              embedding_model=embedding_model,
                              top_k=top_k)
        model_status = """模型已成功重新加载，可以开始对话，或从右侧选择模式后开始对话"""
        logger.info(model_status)
    except Exception as e:
        logger.error(e)
        model_status = """模型未成功重新加载，请到页面左上角"模型配置"选项卡中重新选择后点击"加载模型"按钮"""
        logger.info(model_status)
    return history + [[None, model_status]]


def get_vector_store(vs_id, files, sentence_size, history, one_conent, one_content_segmentation):
    vs_path = os.path.join(KB_ROOT_PATH, vs_id, "vector_store")
    filelist = []
    if local_doc_qa.llm_model_chain and local_doc_qa.embeddings:
        if isinstance(files, list):
            for file in files:
                filename = os.path.split(file.name)[-1]
                shutil.move(file.name, os.path.join(KB_ROOT_PATH, vs_id, "content", filename))
                filelist.append(os.path.join(KB_ROOT_PATH, vs_id, "content", filename))
            vs_path, loaded_files = local_doc_qa.init_knowledge_vector_store(filelist, vs_path, sentence_size)
        else:
            vs_path, loaded_files = local_doc_qa.one_knowledge_add(vs_path, files, one_conent, one_content_segmentation,
                                                                   sentence_size)
        if len(loaded_files):
            file_status = f"已添加 {'、'.join([os.path.split(i)[-1] for i in loaded_files if i])} 内容至知识库，并已加载知识库，请开始提问"
        else:
            file_status = "文件未成功加载，请重新上传文件"
    else:
        file_status = "模型未完成加载，请先在加载模型后再导入文件"
        vs_path = None
    logger.info(file_status)
    return vs_path, None, history + [[None, file_status]], \
           gr.update(choices=local_doc_qa.list_file_from_vector_store(vs_path) if vs_path else [])


def change_vs_name_input(vs_id, history):
    if vs_id == "新建知识库":
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), None, history, \
               gr.update(choices=[]), gr.update(visible=False)
    else:
        vs_path = os.path.join(KB_ROOT_PATH, vs_id, "vector_store")
        if "index.faiss" in os.listdir(vs_path):
            file_status = f"已加载知识库{vs_id}，请开始提问"
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), \
                   vs_path, history + [[None, file_status]], \
                   gr.update(choices=local_doc_qa.list_file_from_vector_store(vs_path), value=[]), \
                   gr.update(visible=True)
        else:
            file_status = f"已选择知识库{vs_id}，当前知识库中未上传文件，请先上传文件后，再开始提问"
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), \
                   vs_path, history + [[None, file_status]], \
                   gr.update(choices=[], value=[]), gr.update(visible=True, value=[])


knowledge_base_test_mode_info = ("【注意】\n\n"
                                 "1. 您已进入知识库测试模式，您输入的任何对话内容都将用于进行知识库查询，"
                                 "并仅输出知识库匹配出的内容及相似度分值和及输入的文本源路径，查询的内容并不会进入模型查询。\n\n"
                                 "2. 知识相关度 Score 经测试，建议设置为 500 或更低，具体设置情况请结合实际使用调整。"
                                 """3. 使用"添加单条数据"添加文本至知识库时，内容如未分段，则内容越多越会稀释各查询内容与之关联的score阈值。\n\n"""
                                 "4. 单条内容长度建议设置在100-150左右。\n\n"
                                 "5. 本界面用于知识入库及知识匹配相关参数设定，但当前版本中，"
                                 "本界面中修改的参数并不会直接修改对话界面中参数，仍需前往`configs/model_config.py`修改后生效。"
                                 "相关参数将在后续版本中支持本界面直接修改。")


def change_mode(mode, history):
    if mode == "知识库问答":
        return gr.update(visible=True), gr.update(visible=False), history
        # + [[None, "【注意】：您已进入知识库问答模式，您输入的任何查询都将进行知识库查询，然后会自动整理知识库关联内容进入模型查询！！！"]]
    elif mode == "知识库测试":
        return gr.update(visible=True), gr.update(visible=True), [[None,
                                                                   knowledge_base_test_mode_info]]
    else:
        return gr.update(visible=False), gr.update(visible=False), history


def change_chunk_conent(mode, label_conent, history):
    conent = ""
    if "chunk_conent" in label_conent:
        conent = "搜索结果上下文关联"
    elif "one_content_segmentation" in label_conent:  # 这里没用上，可以先留着
        conent = "内容分段入库"

    if mode:
        return gr.update(visible=True), history + [[None, f"【已开启{conent}】"]]
    else:
        return gr.update(visible=False), history + [[None, f"【已关闭{conent}】"]]


def add_vs_name(vs_name, chatbot):
    if vs_name is None or vs_name.strip() == "":
        vs_status = "知识库名称不能为空，请重新填写知识库名称"
        chatbot = chatbot + [[None, vs_status]]
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(
            visible=False), chatbot, gr.update(visible=False)
    elif vs_name in get_vs_list():
        vs_status = "与已有知识库名称冲突，请重新选择其他名称后提交"
        chatbot = chatbot + [[None, vs_status]]
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(
            visible=False), chatbot, gr.update(visible=False)
    else:
        # 新建上传文件存储路径
        if not os.path.exists(os.path.join(KB_ROOT_PATH, vs_name, "content")):
            os.makedirs(os.path.join(KB_ROOT_PATH, vs_name, "content"))
        # 新建向量库存储路径
        if not os.path.exists(os.path.join(KB_ROOT_PATH, vs_name, "vector_store")):
            os.makedirs(os.path.join(KB_ROOT_PATH, vs_name, "vector_store"))
        vs_status = f"""已新增知识库"{vs_name}",将在上传文件并载入成功后进行存储。请在开始对话前，先完成文件上传。 """
        chatbot = chatbot + [[None, vs_status]]
        return gr.update(visible=True, choices=get_vs_list(), value=vs_name), gr.update(
            visible=False), gr.update(visible=False), gr.update(visible=True), chatbot, gr.update(visible=True)


# 自动化加载固定文件间中文件
def reinit_vector_store(vs_id, history):
    try:
        shutil.rmtree(os.path.join(KB_ROOT_PATH, vs_id, "vector_store"))
        vs_path = os.path.join(KB_ROOT_PATH, vs_id, "vector_store")
        sentence_size = gr.Number(value=SENTENCE_SIZE, precision=0,
                                  label="文本入库分句长度限制",
                                  interactive=True, visible=True)
        vs_path, loaded_files = local_doc_qa.init_knowledge_vector_store(os.path.join(KB_ROOT_PATH, vs_id, "content"),
                                                                         vs_path, sentence_size)
        model_status = """知识库构建成功"""
    except Exception as e:
        logger.error(e)
        model_status = """知识库构建未成功"""
        logger.info(model_status)
    return history + [[None, model_status]]


def refresh_vs_list(req: gr.Request):
    # 找到用户对应文件
    user_id = req.username
    user_path = USER_INFO_PATH
    file_list = os.listdir(user_path)
    user_file_id = user_id + ".txt"
    user_file_path = os.path.join(USER_INFO_PATH, user_file_id)
    user_info = {}
    print(f"user_file_path = {user_file_path}")
    print(f"file_list = {file_list}")

    if user_file_id not in file_list:
        # 没有用户对应信息，创建新的文件，拥有一个默认会话
        default_conv = {"id": user_id,
                        "info": [["新建会话", [[None, "对话新建成功"]]]]
                        }
        # 默认会话创建文件
        with open(user_file_path, "w+", encoding="utf-8") as file:
            json.dump(default_conv, file, ensure_ascii=False, indent=4)
        # 默认会话赋值给用户信息
        user_info = default_conv
    else:
        # 已经存在用户信息，直接加载
        try:
            with open(user_file_path, "r", encoding="utf-8") as file:
                user_info = json.load(file)
        except:
            # 没有用户对应信息，创建新的文件，拥有一个默认会话
            default_conv = {"id": user_id,
                            "info": [["新建会话", [[None, "对话新建成功"]]]]
                            }
            # 默认会话创建文件
            with open(user_file_path, "w+", encoding="utf-8") as file:
                json.dump(default_conv, file, ensure_ascii=False, indent=4)
            # 默认会话赋值给用户信息
            user_info = default_conv
    chat_list = [x[0] for x in user_info["info"]]
    return user_info, \
           gr.update(choices=chat_list, value=chat_list[0]), \
           gr.update(value=user_info["info"][0][1]), \
           gr.update(choices=get_vs_list()), \
           gr.update(choices=get_vs_list())



def get_user_info(req: gr.Request):
    # 找到用户对应文件
    user_id = req.username
    user_path = USER_INFO_PATH
    file_list = os.listdir(user_path)
    user_file_id = user_id + ".txt"
    user_file_path = os.path.join(USER_INFO_PATH, user_file_id)
    user_info = {}
    print(f"user_file_path = {user_file_path}")
    print(f"file_list = {file_list}")

    if user_file_id not in file_list:
        # 没有用户对应信息，创建新的文件，拥有一个默认会话
        default_conv = {"id": user_id,
                        "info": [["新建会话", [[None, "对话新建成功"]]]]
                        }
        # 默认会话创建文件
        with open(user_file_path, "w+", encoding="utf-8") as file:
            json.dump(default_conv, file, ensure_ascii=False, indent=4)
        # 默认会话赋值给用户信息
        user_info = default_conv
    else:
        # 已经存在用户信息，直接加载
        try:
            with open(user_file_path, "r", encoding="utf-8") as file:
                user_info = json.load(file)
        except:
            # 没有用户对应信息，创建新的文件，拥有一个默认会话
            default_conv = {"id": user_id,
                            "info": [["新建会话", [[None, "对话新建成功"]]]]
                            }
            # 默认会话创建文件
            with open(user_file_path, "w+", encoding="utf-8") as file:
                json.dump(default_conv, file, ensure_ascii=False, indent=4)
            # 默认会话赋值给用户信息
            user_info = default_conv
    chat_list = [x[0] for x in user_info["info"]]
    return user_info, \
           gr.update(choices=chat_list, value=chat_list[0]), \
           gr.update(value=user_info["info"][0][1]), \



def delete_file(vs_id, files_to_delete, chatbot):
    vs_path = os.path.join(KB_ROOT_PATH, vs_id, "vector_store")
    content_path = os.path.join(KB_ROOT_PATH, vs_id, "content")
    docs_path = [os.path.join(content_path, file) for file in files_to_delete]
    status = local_doc_qa.delete_file_from_vector_store(vs_path=vs_path,
                                                        filepath=docs_path)
    if "fail" not in status:
        for doc_path in docs_path:
            if os.path.exists(doc_path):
                os.remove(doc_path)
    rested_files = local_doc_qa.list_file_from_vector_store(vs_path)
    if "fail" in status:
        vs_status = "文件删除失败。"
    elif len(rested_files) > 0:
        vs_status = "文件删除成功。"
    else:
        vs_status = f"文件删除成功，知识库{vs_id}中无已上传文件，请先上传文件后，再开始提问。"
    logger.info(",".join(files_to_delete) + vs_status)
    chatbot = chatbot + [[None, vs_status]]
    return gr.update(choices=local_doc_qa.list_file_from_vector_store(vs_path), value=[]), chatbot


def delete_vs(vs_id, chatbot):
    try:
        shutil.rmtree(os.path.join(KB_ROOT_PATH, vs_id))
        status = f"成功删除知识库{vs_id}"
        logger.info(status)
        chatbot = chatbot + [[None, status]]
        return gr.update(choices=get_vs_list(), value=get_vs_list()[0]), gr.update(visible=True), gr.update(
            visible=True), \
               gr.update(visible=False), chatbot, gr.update(visible=False)
    except Exception as e:
        logger.error(e)
        status = f"删除知识库{vs_id}失败"
        chatbot = chatbot + [[None, status]]
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), \
               gr.update(visible=True), chatbot, gr.update(visible=True)


def switch_user_mode():
    return gr.update(visible=False), gr.update(visible=True)


def switch_admin_mode():
    return gr.update(visible=True), gr.update(visible=False)


def regenerate(req:gr.Request, vs_path, history, mode, admin_flag, chatbot_info, cur_chat_id,
               score_threshold=VECTOR_SEARCH_SCORE_THRESHOLD,vector_search_top_k=VECTOR_SEARCH_TOP_K,
               chunk_conent: bool = True, chunk_size=CHUNK_SIZE, streaming: bool = STREAMING):
    print(f"history={history}")
    global final_history
    # global last_query
    last_query = history[-1][0]
    exception_flag = False
    try:
        generator = local_doc_qa.get_knowledge_based_answer(query=last_query, vs_path=vs_path, chat_history=history,
                                                            streaming=streaming, regenerate=True)
        while True:
            # generator = local_doc_qa.get_knowledge_based_answer(query=query, vs_path=vs_path, chat_history=history, streaming=streaming)
            resp, history = next(generator)
            if admin_flag:
                source = "\n\n"
                source += "".join(
                    [f"""<details> <summary>出处 [{i + 1}] {os.path.split(doc.metadata["source"])[-1]}</summary>\n"""
                     f"""{doc.page_content}\n"""
                     f"""</details>"""
                     for i, doc in
                     enumerate(resp["source_documents"])])
                history[-1][-1] += source
            # print(f"history:{history}")
            final_history = history
            yield history, gr.update(visible=True), gr.update(visible=False), gr.update(), chatbot_info
    except StopIteration:
        print("regenerate StopIteration")
        chat_list, chat_title, chatbot_info = _answer_record(req.username, final_history[-1], chatbot_info, cur_chat_id)
        yield final_history, gr.update(visible=False), gr.update(visible=True), \
            gr.update(choices=chat_list, value=chat_title), chatbot_info
    except GeneratorExit:
        print("regenerate GeneratorExit")
        exception_flag = True
        # _answer_record(req.username, final_history[-1], chatbot_info, cur_chat_id)
    finally:
        print("regenerate finally")
        if not exception_flag:
            chat_list, chat_title, chatbot_info = _answer_record(req.username, final_history[-1], chatbot_info, cur_chat_id)
            yield final_history, gr.update(visible=False), gr.update(visible=True), \
                gr.update(choices=chat_list, value=chat_title), chatbot_info


def get_hotquestion():
    with open("hotquestion.txt", "r") as f:
        content = json.load(f)
    source = """<p class="bold-text">问题推荐</p><br>"""
    source += "".join(
        [f"""<details> <summary>{item["question"]}</summary>\n"""
         f"""{item["answer"]}\n"""
         f"""</details><br>"""
         for item in content["content"]])
    return source


def terminal_conv():
    return gr.update(visible=False), gr.update(visible=True)


def get_user_info(req: gr.Request):
    # 找到用户对应文件
    user_id = req.username
    user_path = USER_INFO_PATH
    file_list = os.listdir(user_path)
    user_file_id = user_id + ".txt"
    user_file_path = os.path.join(USER_INFO_PATH, user_file_id)
    user_info = {}

    if user_file_id not in file_list:
        # 没有用户对应信息，创建新的文件，拥有一个默认会话
        default_conv = {"id": user_id,
                        "info": [["新建会话", [[None, "对话新建成功"]]]]
                        }
        # 默认会话创建文件
        with open(user_file_path, "w+", encoding="utf-8") as file:
            json.dump(default_conv, file, ensure_ascii=False, indent=4)
        # 默认会话赋值给用户信息
        user_info = default_conv
    else:
        # 已经存在用户信息，直接加载
        try:
            with open(user_file_path, "r", encoding="utf-8") as file:
                user_info = json.load(file)
        except:
            # 没有用户对应信息，创建新的文件，拥有一个默认会话
            default_conv = {"id": user_id,
                            "info": [["新建会话", [[None, "对话新建成功"]]]]
                            }
            # 默认会话创建文件
            with open(user_file_path, "w+", encoding="utf-8") as file:
                json.dump(default_conv, file, ensure_ascii=False, indent=4)
            # 默认会话赋值给用户信息
            user_info = default_conv
    chat_list = [x[0] for x in user_info["info"]]
    return user_info, \
           gr.update(choices=chat_list, value=chat_list[0]), \
           gr.update(value=user_info["info"][0][1])


def record_user_info(user_id, cur_id=None, chat_id=None, delete_id=None, mode="", new_chat=None):
    if new_chat is None:
        new_chat = []
    if mode == "create":
        cur_id = ""
        chat_id = "新建会话"
    elif mode == "delete":
        cur_id = delete_id
        chat_id = ""
    print(f"enter: user_id={user_id}, cur_id={cur_id}, chat_id={chat_id}, new_chat={new_chat}")

    user_file_id = user_id + ".txt"
    user_file_path = os.path.join(USER_INFO_PATH, user_file_id)
    with file_lock:
        try:
            # 获取用户对话信息
            with open(user_file_path, "r", encoding="utf-8") as file:
                user_info = json.load(file)
            # 如果cur_id=="" & chat_id=="新建会话"表示新建会话
            if cur_id == "" and chat_id == "新建会话":
                chat_list = [x[0] for x in user_info["info"]]
                if "新建会话" not in chat_list:
                    print("新建会话前:", user_info["info"])
                    user_info["info"].append(["新建会话", [[None, "对话新建成功"]]])
                    print("新建会话后:", user_info["info"])
            # 如果cur_id存在 & chat_id==""，表示删除会话
            elif cur_id and not chat_id:
                for i, chat in enumerate(user_info["info"]):
                    if chat[0] == cur_id:
                        del user_info["info"][i]
                        # 如果删除后没有对话，新建默认会话
                        if not user_info["info"]:
                            user_info["info"] = [["新建会话", [[None, "对话新建成功"]]]]
            # 将新对话append到会话list
            else:
                for i, chat in enumerate(user_info["info"]):
                    if chat[0] == cur_id:
                        print(user_info["info"])
                        user_info["info"][i][0] = chat_id
                        user_info["info"][i][1].append(new_chat)
                        print(user_info["info"])
            with open(user_file_path, "w+", encoding="utf-8") as file:
                json.dump(user_info, file, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"Error processing file: {str(e)}")


def change_chat(chatbot_info, cur_chat_id):
    if chatbot_info:
        for chat in chatbot_info["info"]:
            if chat[0] == cur_chat_id:
                return chat[1]


def delete_chat(req: gr.Request, chatbot_info, cur_chat_id):
    if chatbot_info:
        for i, chat in enumerate(chatbot_info["info"]):
            if chat[0] == cur_chat_id:
                del chatbot_info["info"][i]
    cur_chat_list = [x[0] for x in chatbot_info["info"]]

    # 如果全部删除后当前会话列表为空，创建一个默认会话窗口
    if not cur_chat_list:
        cur_chat_list = ["新建会话"]
        chatbot_info["info"] = [["新建会话", [[None, "对话新建成功"]]]]

    # 更新记录
    record_user_info(req.username, mode="delete", delete_id=cur_chat_id)

    value = cur_chat_list[0]

    return gr.update(choices=cur_chat_list, value=value), \
           chatbot_info["info"][0][1], \
           chatbot_info


def create_chat(req: gr.Request, chatbot_info):
    chat_list = [x[0] for x in chatbot_info["info"]]
    if "新建会话" not in chat_list:
        chatbot_info["info"].append(["新建会话", [[None, "对话新建成功"]]])
        chat_list.append("新建会话")
        # 更新记录
        record_user_info(req.username, mode="create")
        return gr.update(choices=chat_list, value="新建会话"), [[None, "对话新建成功"]], chatbot_info
    else:
        return gr.update(choices=chat_list, value="新建会话"), [[None, "对话新建成功"]], chatbot_info


# 截取会话名，过长截取
def process_title(string):
    count = 0  # 计数器，用于记录字符数
    index = 0  # 索引，用于遍历字符串
    while count < 16 and index < len(string):
        char = string[index]
        if ord(char) >= 0x4e00 and ord(char) <= 0x9fff:
            # 如果是汉字（Unicode范围）
            count += 2
        else:
            # 如果是非汉字字符
            count += 1
        index += 1
    # 截取字符串的前八个字符
    result = string[:index]
    return result


block_css = """.importantButton {
    background: linear-gradient(45deg, #7e0570,#5d1c99, #6e00ff) !important;
    border: none !important;
}
.importantButton:hover {
    background: linear-gradient(45deg, #ff00e0,#8500ff, #6e00ff) !important;
    border: none !important;
}
details {
  border: 1px solid #aaa;
  border-radius: 4px;
  padding: 0.5em 0.5em 0;
}

summary {
  font-weight: bold;
  margin: -0.5em -0.5em 0;
  padding: 0.5em;
}

details[open] {
  padding: 0.5em;
}

details[open] summary {
  border-bottom: 1px solid #aaa;
  margin-bottom: 0.5em;
}
.bold-text {
    font-weight: bold;
    font-size: 130%; 
}

.wrap.svelte-1p9xokt {
  flex-direction:column
}

.model {
    display: flex;
    position: fixed;
    z-index: 1;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    overflow: auto;
    background-color: rgba(0,0,0,0.4);
}

        /* 弹出框内容的样式 */
.modal-content {
    background-color: #fefefe;
    margin: 15% auto;
    padding: 20px;
    border: 1px solid #888;
    width: 80%;
}
"""

webui_title = """
# 🎉计算机网络问答系统🎉
"""
default_vs = get_vs_list()[0] if len(get_vs_list()) > 1 else "为空"
init_message_user = f"""欢迎使用计算机网络问答系统！
"""
init_message = f"""欢迎使用计算机网络问答系统！

请在右侧切换模式，目前支持直接与 LLM 模型对话或基于本地知识库问答。

知识库问答模式，选择知识库名称后，即可开始问答，当前知识库{default_vs}，如有需要可以在选择知识库名称后上传文件/文件夹至知识库。
"""

# 初始化消息
model_status_s = init_model()
hotquestion = get_hotquestion()
# model_status_user = init_model()

default_theme_args = dict(
    font=["Source Sans Pro", 'ui-sans-serif', 'system-ui', 'sans-serif'],
    font_mono=['IBM Plex Mono', 'ui-monospace', 'Consolas', 'monospace'],
)

with gr.Blocks(css=block_css, theme=gr.themes.Default(**default_theme_args)) as admin:
    vs_path, file_status, model_status, admin_flag, chatbot_info = gr.State(
        os.path.join(KB_ROOT_PATH, "新建知识库", "vector_store") if len(get_vs_list()) > 1 else ""), \
                                                                   gr.State(""), \
                                                                   gr.State(model_status_s), \
                                                                   gr.State(True), \
                                                                   gr.State({})
    gr.Markdown(webui_title)
    with gr.Row():
        # with gr.Column(scale=1):
        #     user_mode = gr.Button(value="用户模式")
        #     admin_mode = gr.Button(value="管理模式")
        with gr.Column(scale=10) as admin_window:
            # admin_flag = True
            with gr.Tab("对话") as dh_tab:
                with gr.Row():
                    with gr.Column(scale=10) as select_bar:
                        chat_list = gr.Radio(label="对话列表", elem_classes=".my_div")
                        create_chat_btn = gr.Button("创建会话")
                        delete_chat_btn = gr.Button("删除会话")
                    with gr.Column(scale=50) as chatbot_column:
                        chatbot = gr.Chatbot([[None, init_message], [None, model_status.value]],
                                             elem_id="chat-box",
                                             show_label=False).style(height=750)
                        query = gr.Textbox(show_label=False,
                                           placeholder="请输入提问内容，按回车进行提交").style(container=False)
                        terminal_button = gr.Button("停止生成", visible=False)
                        regenerate_button = gr.Button("重新生成", visible=False)
                    with gr.Column(scale=20) as vs_column:
                        mode = gr.Radio(["LLM 对话", "知识库问答", "Bing搜索问答"],
                                        label="请选择使用模式",
                                        value="知识库问答", )
                        knowledge_set = gr.Accordion("知识库设定", visible=False)
                        vs_setting = gr.Accordion("配置知识库")
                        mode.change(fn=change_mode,
                                    inputs=[mode, chatbot],
                                    outputs=[vs_setting, knowledge_set, chatbot])
                        # admin_flag = gr.Checkbox(value=True,
                        # visible=False)
                        with vs_setting:
                            vs_refresh = gr.Button("更新已有知识库选项")
                            select_vs = gr.Dropdown(get_vs_list(),
                                                    label="请选择要加载的知识库",
                                                    interactive=True,
                                                    value=get_vs_list()[0] if len(get_vs_list()) > 0 else None
                                                    )
                            vs_name = gr.Textbox(label="请输入新建知识库名称，当前知识库命名暂不支持中文",
                                                 lines=1,
                                                 interactive=True,
                                                 visible=True)
                            vs_add = gr.Button(value="添加至知识库选项", visible=True)
                            vs_delete = gr.Button("删除本知识库", visible=False)
                            file2vs = gr.Column(visible=False)
                            with file2vs:
                                # load_vs = gr.Button("加载知识库")
                                gr.Markdown("向知识库中添加文件")
                                sentence_size = gr.Number(value=SENTENCE_SIZE, precision=0,
                                                          label="文本入库分句长度限制",
                                                          interactive=True, visible=True)
                                with gr.Tab("上传文件"):
                                    files = gr.File(label="添加文件",
                                                    file_types=['.txt', '.md', '.docx', '.pdf', '.png', '.jpg', ".csv"],
                                                    file_count="multiple",
                                                    show_label=False)
                                    load_file_button = gr.Button("上传文件并加载知识库")
                                with gr.Tab("上传文件夹"):
                                    folder_files = gr.File(label="添加文件",
                                                           file_count="directory",
                                                           show_label=False)
                                    load_folder_button = gr.Button("上传文件夹并加载知识库")
                                with gr.Tab("删除文件"):
                                    files_to_delete = gr.CheckboxGroup(choices=[],
                                                                       label="请从知识库已有文件中选择要删除的文件",
                                                                       interactive=True)
                                    delete_file_button = gr.Button("从知识库中删除选中文件")
                            vs_refresh.click(fn=refresh_vs_list,
                                             inputs=[],
                                             outputs=select_vs)
                            vs_add.click(fn=add_vs_name,
                                         inputs=[vs_name, chatbot],
                                         outputs=[select_vs, vs_name, vs_add, file2vs, chatbot, vs_delete])
                            vs_delete.click(fn=delete_vs,
                                            inputs=[select_vs, chatbot],
                                            outputs=[select_vs, vs_name, vs_add, file2vs, chatbot, vs_delete])
                            select_vs.change(fn=change_vs_name_input,
                                             inputs=[select_vs, chatbot],
                                             outputs=[vs_name, vs_add, file2vs, vs_path, chatbot, files_to_delete,
                                                      vs_delete])
                            load_file_button.click(get_vector_store,
                                                   show_progress=True,
                                                   inputs=[select_vs, files, sentence_size, chatbot, vs_add, vs_add],
                                                   outputs=[vs_path, files, chatbot, files_to_delete], )
                            load_folder_button.click(get_vector_store,
                                                     show_progress=True,
                                                     inputs=[select_vs, folder_files, sentence_size, chatbot, vs_add,
                                                             vs_add],
                                                     outputs=[vs_path, folder_files, chatbot, files_to_delete], )
                            flag_csv_logger.setup([query, vs_path, chatbot, mode], "flagged")
                            predict_event = query.submit(get_answer,
                                                         [query, vs_path, chatbot, mode, admin_flag,
                                                                chatbot_info, chat_list],
                                                         [chatbot, query, terminal_button,
                                                                regenerate_button, chat_list, chatbot_info])
                            regenerate_event = regenerate_button.click(regenerate,
                                                                       show_progress=True,
                                                                       inputs=[vs_path, chatbot, mode, admin_flag,
                                                                               chatbot_info, chat_list],
                                                                       outputs=[chatbot, terminal_button,
                                                                                regenerate_button, chat_list, chatbot_info])
                            terminal_button.click(terminal_conv,
                                                  inputs=[],
                                                  outputs=[terminal_button, regenerate_button],
                                                  cancels=[predict_event, regenerate_event])
                            delete_file_button.click(delete_file,
                                                     show_progress=True,
                                                     inputs=[select_vs, files_to_delete, chatbot],
                                                     outputs=[files_to_delete, chatbot])
                            chat_list.change(fn=change_chat,
                                             inputs=[chatbot_info, chat_list],
                                             outputs=[chatbot])
                            create_chat_btn.click(fn=create_chat,
                                                  inputs=[chatbot_info],
                                                  outputs=[chat_list, chatbot, chatbot_info])
                            delete_chat_btn.click(fn=delete_chat,
                                                  inputs=[chatbot_info, chat_list],
                                                  outputs=[chat_list, chatbot, chatbot_info])
            with gr.Tab("知识库测试 Beta") as beta_tab:
                with gr.Row():
                    with gr.Column(scale=10):
                        chatbot = gr.Chatbot([[None, knowledge_base_test_mode_info]],
                                             elem_id="chat-box",
                                             show_label=False).style(height=750)
                        query = gr.Textbox(show_label=False,
                                           placeholder="请输入提问内容，按回车进行提交").style(container=False)
                    with gr.Column(scale=5):
                        mode = gr.Radio(["知识库测试"],  # "知识库问答",
                                        label="请选择使用模式",
                                        value="知识库测试",
                                        visible=False)
                        knowledge_set = gr.Accordion("知识库设定", visible=True)
                        vs_setting = gr.Accordion("配置知识库", visible=True)
                        mode.change(fn=change_mode,
                                    inputs=[mode, chatbot],
                                    outputs=[vs_setting, knowledge_set, chatbot])
                        with knowledge_set:
                            score_threshold = gr.Number(value=VECTOR_SEARCH_SCORE_THRESHOLD,
                                                        label="知识相关度 Score 阈值，分值越低匹配度越高",
                                                        precision=0,
                                                        interactive=True)
                            vector_search_top_k = gr.Number(value=VECTOR_SEARCH_TOP_K, precision=0,
                                                            label="获取知识库内容条数", interactive=True)
                            chunk_conent = gr.Checkbox(value=False,
                                                       label="是否启用上下文关联",
                                                       interactive=True)
                            chunk_sizes = gr.Number(value=CHUNK_SIZE, precision=0,
                                                    label="匹配单段内容的连接上下文后最大长度",
                                                    interactive=True, visible=False)
                            chunk_conent.change(fn=change_chunk_conent,
                                                inputs=[chunk_conent, gr.Textbox(value="chunk_conent", visible=False),
                                                        chatbot],
                                                outputs=[chunk_sizes, chatbot])
                        with vs_setting:
                            vs_refresh = gr.Button("更新已有知识库选项")
                            select_vs_test = gr.Dropdown(get_vs_list(),
                                                         label="请选择要加载的知识库",
                                                         interactive=True,
                                                         value=get_vs_list()[0] if len(get_vs_list()) > 0 else None)
                            vs_name = gr.Textbox(label="请输入新建知识库名称，当前知识库命名暂不支持中文",
                                                 lines=1,
                                                 interactive=True,
                                                 visible=True)
                            vs_add = gr.Button(value="添加至知识库选项", visible=True)
                            file2vs = gr.Column(visible=False)
                            with file2vs:
                                # load_vs = gr.Button("加载知识库")我在函数外修改这个bool类型的参数，
                                gr.Markdown("向知识库中添加单条内容或文件")
                                sentence_size = gr.Number(value=SENTENCE_SIZE, precision=0,
                                                          label="文本入库分句长度限制",
                                                          interactive=True, visible=True)
                                with gr.Tab("上传文件"):
                                    files = gr.File(label="添加文件",
                                                    file_types=['.txt', '.md', '.docx', '.pdf'],
                                                    file_count="multiple",
                                                    show_label=False
                                                    )
                                    load_file_button = gr.Button("上传文件并加载知识库")
                                with gr.Tab("上传文件夹"):
                                    folder_files = gr.File(label="添加文件",
                                                           # file_types=['.txt', '.md', '.docx', '.pdf'],
                                                           file_count="directory",
                                                           show_label=False)
                                    load_folder_button = gr.Button("上传文件夹并加载知识库")
                                with gr.Tab("添加单条内容"):
                                    one_title = gr.Textbox(label="标题", placeholder="请输入要添加单条段落的标题",
                                                           lines=1)
                                    one_conent = gr.Textbox(label="内容", placeholder="请输入要添加单条段落的内容",
                                                            lines=5)
                                    one_content_segmentation = gr.Checkbox(value=True, label="禁止内容分句入库",
                                                                           interactive=True)
                                    load_conent_button = gr.Button("添加内容并加载知识库")
                            # 将上传的文件保存到content文件夹下,并更新下拉框
                            vs_refresh.click(fn=refresh_vs_list,
                                             inputs=[],
                                             outputs=select_vs_test)
                            vs_add.click(fn=add_vs_name,
                                         inputs=[vs_name, chatbot],
                                         outputs=[select_vs_test, vs_name, vs_add, file2vs, chatbot])
                            select_vs_test.change(fn=change_vs_name_input,
                                                  inputs=[select_vs_test, chatbot],
                                                  outputs=[vs_name, vs_add, file2vs, vs_path, chatbot])
                            load_file_button.click(get_vector_store,
                                                   show_progress=True,
                                                   inputs=[select_vs_test, files, sentence_size, chatbot, vs_add,
                                                           vs_add],
                                                   outputs=[vs_path, files, chatbot], )
                            load_folder_button.click(get_vector_store,
                                                     show_progress=True,
                                                     inputs=[select_vs_test, folder_files, sentence_size, chatbot,
                                                             vs_add,
                                                             vs_add],
                                                     outputs=[vs_path, folder_files, chatbot], )
                            load_conent_button.click(get_vector_store,
                                                     show_progress=True,
                                                     inputs=[select_vs_test, one_title, sentence_size, chatbot,
                                                             one_conent, one_content_segmentation],
                                                     outputs=[vs_path, files, chatbot], )
                            flag_csv_logger.setup([query, vs_path, chatbot, mode], "flagged")
                            query.submit(get_answer,
                                         [query, vs_path, chatbot, mode, score_threshold, vector_search_top_k,
                                          chunk_conent,
                                          chunk_sizes],
                                         [chatbot, query])
            with gr.Tab("模型配置") as pz_tab:
                llm_model = gr.Radio(llm_model_dict_list,
                                     label="LLM 模型",
                                     value=LLM_MODEL,
                                     interactive=True)
                no_remote_model = gr.Checkbox(shared.LoaderCheckPoint.no_remote_model,
                                              label="加载本地模型",
                                              interactive=True)

                llm_history_len = gr.Slider(0, 10,
                                            value=LLM_HISTORY_LEN,
                                            step=1,
                                            label="LLM 对话轮数",
                                            interactive=True)
                use_ptuning_v2 = gr.Checkbox(USE_PTUNING_V2,
                                             label="使用p-tuning-v2微调过的模型",
                                             interactive=True)
                use_lora = gr.Checkbox(USE_LORA,
                                       label="使用lora微调的权重",
                                       interactive=True)
                embedding_model = gr.Radio(embedding_model_dict_list,
                                           label="Embedding 模型",
                                           value=EMBEDDING_MODEL,
                                           interactive=True)
                top_k = gr.Slider(1, 20, value=VECTOR_SEARCH_TOP_K, step=1,
                                  label="向量匹配 top k", interactive=True)
                load_model_button = gr.Button("重新加载模型")
                load_model_button.click(reinit_model, show_progress=True,
                                        inputs=[llm_model, embedding_model, llm_history_len, no_remote_model,
                                                use_ptuning_v2,
                                                use_lora, top_k, chatbot], outputs=chatbot)
                # load_knowlege_button = gr.Button("重新构建知识库")
                # load_knowlege_button.click(reinit_vector_store, show_progress=True,
                #                            inputs=[select_vs, chatbot], outputs=chatbot)

        # user_mode.click(fn=switch_user_mode,
        #                 outputs=[admin_window, user_window])
        # admin_mode.click(fn=switch_admin_mode,
        #                  outputs=[admin_window, user_window])
    admin.load(
        fn=refresh_vs_list,
        inputs=None,
        outputs=[chatbot_info, chat_list, chatbot, select_vs, select_vs_test],
        queue=True,
        show_progress=False,
    )

with gr.Blocks(css=block_css, theme=gr.themes.Default(**default_theme_args)) as user:
    vs_path, file_status, model_status, admin_flag, chatbot_info = gr.State(
        os.path.join(KB_ROOT_PATH, "jiwang", "vector_store") if len(get_vs_list()) > 1 else ""), \
                                                                   gr.State(""), \
                                                                   gr.State(model_status_s), \
                                                                   gr.State(False), \
                                                                   gr.State({})
    gr.Markdown(webui_title)
    with gr.Row():
        with gr.Column(scale=10) as select_bar:
            chat_list = gr.Radio(label="对话列表", elem_classes=".my_div")
            create_chat_btn = gr.Button("创建会话")
            delete_chat_btn = gr.Button("删除会话")
        with gr.Column(scale=60) as chatbot_bar:
            with gr.Tab("对话"):
                with gr.Row():
                    with gr.Column(scale=10):
                        chatbot = gr.Chatbot([],
                                             elem_id="chat-box",
                                             show_label=False).style(height=750)
                        query = gr.Textbox(show_label=False,
                                           placeholder="请输入提问内容，按回车进行提交").style(container=False)
                        terminal_button = gr.Button("停止生成", visible=False)
                        regenerate_button = gr.Button("重新生成", visible=False)
                        mode = gr.Radio(["LLM 对话", "知识库问答", "Bing搜索问答"],
                                        label="请选择使用模式",
                                        value="知识库问答",
                                        visible=False)
                        predict_event_user = query.submit(get_answer,
                                                               [query, vs_path, chatbot, mode, admin_flag,
                                                                chatbot_info, chat_list],
                                                               [chatbot, query, terminal_button,
                                                                regenerate_button, chat_list, chatbot_info])
                        regenerate_event_user = regenerate_button.click(regenerate,
                                                                             show_progress=True,
                                                                             inputs=[vs_path, chatbot, mode,
                                                                                     admin_flag, chatbot_info,
                                                                                     chat_list],
                                                                             outputs=[chatbot,
                                                                                      terminal_button,
                                                                                      regenerate_button,
                                                                                      chat_list,
                                                                                      chatbot_info])
                        terminal_button.click(terminal_conv,
                                                   inputs=[],
                                                   outputs=[terminal_button, regenerate_button],
                                                   cancels=[predict_event_user, regenerate_event_user])
                        chat_list.change(fn=change_chat,
                                         inputs=[chatbot_info, chat_list],
                                         outputs=[chatbot])
                        create_chat_btn.click(fn=create_chat,
                                              inputs=[chatbot_info],
                                              outputs=[chat_list, chatbot, chatbot_info])
                        delete_chat_btn.click(fn=delete_chat,
                                              inputs=[chatbot_info, chat_list],
                                              outputs=[chat_list, chatbot, chatbot_info])
    user.load(
        fn=get_user_info,
        inputs=None,
        outputs=[chatbot_info, chat_list, chatbot],
        queue=True,
        show_progress=False,
    )


def launch_block(block, port):
    (block
     .queue(concurrency_count=3)
     .launch(server_name='0.0.0.0',
             server_port=port,
             auth=login_func,
             show_api=False,
             share=False,
             inbrowser=False))


if __name__ == "__main__":
    #thread1 = threading.Thread(target=launch_block, args=(user, 7860))
    #thread2 = threading.Thread(target=launch_block, args=(admin, 7861))

    #thread1.start()
    #thread2.start()

    #thread1.join()
    #thread2.join()
    launch_block(user, 7860)


