from transformers import AutoModelForCausalLM, AutoTokenizer,LlamaTokenizer
import json
import argparse
import torch
from tqdm import tqdm
from utils import *
import os
from vllm import LLM, SamplingParams


boss_reflection_prompt = '''请你扮演一个面试官，根据<应聘者简历>、<岗位描述>、<面试官提问策略>和面试对话，复盘反思，更新面试官提问策略。*用一句话给出策略，要求策略应该是高度抽象，对所有应聘者适用，尽量和之前的策略相同，回答要尽可能简洁*'''
geek_reflection_prompt = '''请你扮演一个应聘者，根据<应聘者简历>、<岗位描述>、<应聘者回答策略>和面试对话，复盘反思，更新应聘者回答策略。*用一句话给出策略，要求策略应该是高度抽象/总结的，对所有面试者适用，尽量和之前的策略相同，回答要尽可能简洁*'''

boss_question_prompt = "请你扮演一个面试官，根据<应聘者简历>、<岗位描述>、和历史面试对话（如果有）向应聘者提问一个技术问题，问题必须和应聘者简历/岗位内容紧密相关，根据项目经历及其相关技能提问，**提问的问题禁止和之前雷同相似，禁止说套话。**，要求尽可能简洁。"
boss_evaluation_prompt = '''请你扮演一个面试官，根据<应聘者简历>、<岗位描述>、和历史面试对话判断该应聘者是否合适这个岗位。注意岗位描述和应聘者的简历是否吻合，你的回复只能是"合适"或者"不合适"。'''

geek_response_prompt = '''请你扮演一个应聘者，根据<应聘者简历>、<岗位描述>、和历史面试对话回答面试官最后的一次提问，**要求回答内容禁止和之前雷同相似，与应聘者简历内容不冲突。**，要求尽可能简短'''

geek_rewrite_prompt = '''请你扮演一个应聘者，根据<应聘者简历>、<岗位描述>、和历史面试对话修改已有应聘者的简历，给出修改后的简历。'''

geek_evaluation_prompt = '''请你扮演一个应聘者，根据<应聘者简历>、<岗位描述>、和历史面试对话判断是否能得到这个岗位。注意岗位描述和应聘者的简历是否吻合，你的回复只能是"合适"或者"不合适"。'''

geek_retrival = '''请你扮演一个应聘者，从<候选岗位描述>中挑选一个内容上最接近<目标岗位描述>的候选，你的回复只能是候选序号'''
boss_retrival = '''请你扮演一个面试官，从<候选简历>中挑选一个内容上最接近<目标简历>的候选，你的回复只能是候选序号'''


sampling_params = SamplingParams(temperature=0.9, top_p=0.6,repetition_penalty=1.1,max_tokens=32768)


def build_prompt_input(query, messages):
    prompt = ""
    for message in messages:
        if message['role'] == 'human':
            prompt += f"""### Human: \n{message['content']}\n\n"""
        elif message['role'] == 'assistant':
            prompt += f"""### Assistant: \n{message['content']}</s>"""
        elif message['role'] == 'system':
            prompt += f"""{message['content']}"""
    prompt += f"""### Human: \n{query}\n\n### Assistant: \n"""
    return prompt

def generate(model, query, tokenizer,
             system_content="你是一个精通人力资源，知晓行各业技能的专家",):
    # msg = [{'role': 'system', 'content': system_content}]
    
    prompt = build_prompt_input(query,msg)
    response = model.generate([prompt],sampling_params)
    return response[0].outputs[0].text

def make_prompt(prompt,resume,JD,dialog,strategy=None,examples=None):
    query = ""
    query += "\n<应聘者简历>\n" + resume + "\n<岗位描述>\n" + JD
    if examples is not None:
        query += "\n<你可以参考的提问回答例子>\n"  if "请你扮演一个面试官" in prompt else "\n<你可以参考的提问回答例子>\n"
        query += examples + '\n'
    if dialog != "":
        query +=  "\n<历史面试对话>\n" +  dialog
    if strategy is not None:
        query += "\n<面试官提问策略>\n" + strategy if "请你扮演一个面试官" in prompt else "\n<应聘者回答策略>\n" + strategy
    query += "\n\n"+ prompt + "\n\n"
    
    # print(query)
    
    return query

def retrival(model,tokenizer,query,candidates,role):
    if role == "boss":
        name = "简历"
        task_prompt = boss_retrival
        system_content = "你是一个来求职的应聘者"
    else:
        name = "岗位描述"
        task_prompt = geek_retrival
        system_content = "你是一个精通人力资源，知晓行各业技能的面试官"
        
    
    prompt = f"\n<目标{name}>\n" + query +'\n\n'
    for idx,candidate in enumerate(candidates):
        prompt += f"\n<候选{name}{idx}>\n" + candidate + '\n'
    prompt += task_prompt +"\n\n你的回答是："
    
    text = generate(model, prompt, tokenizer,system_content)
    idd = 0
    for i in range(1,len(candidates)+1):
        if str(i) in text:
            idd = i
            break
    idd = idd - 1 if idd!=0 else 0
    return idd


def boss_generate(model,tokenizer,boss_dialog,geek_dialog,resume,JD,action="question",boss_strategy=None,examples=None):
    system_content = "你是一个精通人力资源，知晓行各业技能的面试官"
    if action == "question":    
        repeat = True
        dialog = ""
        iters = 3
        for i in range(len(boss_dialog)):
            dialog += "面试官：" + boss_dialog[i] + '\n'
            dialog += "应聘者：" + geek_dialog[i] + '\n'        
        while repeat and iters > 0:
            query = make_prompt(boss_question_prompt,resume,JD,dialog,boss_strategy,examples)
            query += '\n你的一个提问问题是：\n\n'
                
            response = generate(model,query,tokenizer,system_content)
            if (len(boss_dialog)!=0 and response == boss_dialog[-1]) or "对不起" in response:
                repeat = True
            else:
                repeat = False
            iters -= 1
        # print("boss question prompt\n",query)
        # print("boss response answer\n",response)
        

        return response
    
    elif action == "evaluation":
        dialog = "" 
        for i in range(len(boss_dialog)):
            dialog += "面试官：" + boss_dialog[i] + '\n'
            dialog += "应聘者：" + geek_dialog[i] + '\n'
        query = make_prompt(boss_evaluation_prompt,resume,JD,dialog) + '\n\n你的回答是：\n\n' 
        
        response = generate(model,query,tokenizer,system_content)
        return response
    
    elif action == "reflection": #提问内容是否和简历相关，提问内容是否是这份工作最看重的技能，提问的问题是否多样。
        dialog = ""
        for i in range(len(boss_dialog)):
            dialog += "面试官：" + boss_dialog[i] + '\n'
            dialog += "应聘者：" + geek_dialog[i] + '\n'
        
        query = make_prompt(boss_reflection_prompt,resume,JD,dialog,boss_strategy)
        query += '\n\n更新后的策略是：\n\n'
        response = generate(model,query,tokenizer,system_content)
        # print("boss reflection prompt\n",query)
        # print("boss reflection answer\n",response)
        
    
        return response


def geek_generate(model,tokenizer,boss_dialog,geek_dialog,resume,JD,action='question',geek_strategy=None,examples=None):
    system_content = "你是一个来求职的应聘者"
    if action == "response":
        dialog = ""
        for i in range(len(boss_dialog)):
            dialog += "面试官：" + boss_dialog[i] + '\n'
            if i != len(boss_dialog)-1:
                dialog += "应聘者：" + geek_dialog[i] + '\n'
        
        query = make_prompt(geek_response_prompt,resume,JD,dialog,geek_strategy,examples)
        query += '\n\n你的回答是：\n\n'
        
        response = generate(model,query,tokenizer,system_content)
        # print("geek response prompt\n",query)
        # print("geek response answer\n",response)
        
        return response
    
    elif action == "reflection": #回复内容是否和简历相关，回复内容是否简洁准确，回复内容是否突出自己的特点，
        dialog = ""
        for i in range(len(boss_dialog)):
            dialog += "面试官：" + boss_dialog[i] + '\n'
            dialog += "应聘者：" + geek_dialog[i] + '\n'
        
        query = make_prompt(geek_reflection_prompt,resume,JD,dialog,geek_strategy)
        query += '\n\n更新后的策略是：\n\n'
    
        response = generate(model,query,tokenizer,system_content)
        # print("geek reflection prompt\n",query)
        # print("geek reflection answer\n",response)
        
        return response
    
    elif action == "evaluation":
        dialog = "" 
        for i in range(len(boss_dialog)):
            dialog += "面试官：" + boss_dialog[i] + '\n'
            dialog += "应聘者：" + geek_dialog[i] + '\n'
        query = make_prompt(geek_evaluation_prompt,resume,JD,dialog) + '\n\n你的回答是：\n\n' 
        
        response = generate(model,query,tokenizer,system_content)
        return response
    
    pass

def dialog_play(model,tokenizer,resume,JD,dialog_len,geek_strategy,boss_strategy,boss_memory=None,geek_memory=None):
    geek_dialog = []
    boss_dialog = []
    
    if boss_memory is not None:
        example_bid = retrival(model,tokenizer,resume,boss_memory['resume'],'boss')
        boss_examples = boss_memory['dialog'][example_bid]
    else:
        boss_examples = None
        
    if geek_memory is not None:
        example_gid = retrival(model,tokenizer,JD,geek_memory['JD'],'geek')
        geek_examples = geek_memory['dialog'][example_gid]
    else:
        geek_examples = None
        
    #dialog turns determined by boss
    dialog_turns = 0
    while dialog_turns<dialog_len:
        response = boss_generate(model,tokenizer,boss_dialog,geek_dialog,resume,JD,action="question",boss_strategy=boss_strategy,examples=boss_examples)
        # print(response)
        boss_dialog.append(response)
        
        response = geek_generate(model,tokenizer,boss_dialog,geek_dialog,resume,JD,action="response",geek_strategy=geek_strategy,examples=geek_examples)
        # print(response)
        geek_dialog.append(response)
        dialog_turns += 1
        
    return geek_dialog, boss_dialog

def train(args,doc_data,train_sample,dialog_data,model,tokenizer):
    
    strategies = {'geek':{},'boss':{}}
    
    # 只用正样本去反思策略
    print("training boss")
    for gid in tqdm(train_sample['geek']):
        for bid in train_sample['geek'][gid]['pos']: 
            geek_old_strategy = strategies['geek'][gid] if gid in strategies['geek'].keys() else "回复内容应该和岗位所需技能或者工作经历相关，回复内容应简洁准确，回复内容应突出自己的特点。"
            boss_old_strategy = strategies['boss'][bid] if bid in strategies['boss'].keys() else "提问内容应该和岗位所需技能或者工作经历相关，提问内容应是这份工作最需要的，提问的问题应该多样。"
            
            geek_dialog, boss_dialog = dialog_play(model,tokenizer,doc_data['resume'][gid],doc_data['JD'][bid],dialog_data[gid+"-"+bid]['dialog_len'],geek_strategy = geek_old_strategy,boss_strategy = boss_old_strategy)
            geek_strategy = geek_generate(model,tokenizer,boss_dialog,geek_dialog,doc_data['resume'][gid],doc_data['JD'][bid],action='reflection',geek_strategy=geek_old_strategy)
            boss_strategy = boss_generate(model,tokenizer,boss_dialog,geek_dialog,doc_data['resume'][gid],doc_data['JD'][bid],action='reflection',boss_strategy=boss_old_strategy)
            strategies['geek'][gid]=geek_strategy
            strategies['boss'][bid]=boss_strategy
    
    #evaluation
    print("training geek")
    for bid in tqdm(train_sample['boss']):
        for gid in train_sample['boss'][bid]['pos']: 
            geek_old_strategy = strategies['geek'][gid] if gid in strategies['geek'].keys() else "回复内容应该和岗位所需技能或者工作经历相关，回复内容应简洁准确，回复内容应突出自己的特点。"
            boss_old_strategy = strategies['boss'][bid] if bid in strategies['boss'].keys() else "提问内容应该和岗位所需技能或者工作经历相关，提问内容应是这份工作最需要的，提问的问题应该多样。"
            
            geek_dialog, boss_dialog = dialog_play(model,tokenizer,doc_data['resume'][gid],doc_data['JD'][bid],dialog_data[gid+"-"+bid]['dialog_len'],geek_strategy = geek_old_strategy,boss_strategy = boss_old_strategy)
            
            geek_strategy = geek_generate(model,tokenizer,boss_dialog,geek_dialog,doc_data['resume'][gid],doc_data['JD'][bid],action='reflection',geek_strategy=geek_old_strategy)
            boss_strategy = boss_generate(model,tokenizer,boss_dialog,geek_dialog,doc_data['resume'][gid],doc_data['JD'][bid],action='reflection',boss_strategy=boss_old_strategy)
            strategies['geek'][gid]=geek_strategy
            strategies['boss'][bid]=boss_strategy
        
    return strategies

def test(args,doc_data,test_sample,dialog_data,model,tokenizer,strategies,memorys):
    
    predicts = {'geek':{},'boss':{}}
    print("testing geek")
    for gid in tqdm(test_sample['geek']):
        bids = test_sample['geek'][gid]['pos'] + test_sample['geek'][gid]['neg']
        for bid in bids: #测试的时候就不更新策略
            geek_old_strategy = strategies['geek'][gid] if gid in strategies['geek'].keys() else "回复内容应该和岗位所需技能或者工作经历相关，回复内容应简洁准确，回复内容应突出自己的特点。"
            boss_old_strategy = strategies['boss'][bid] if bid in strategies['boss'].keys() else "提问内容应该和岗位所需技能或者工作经历相关，提问内容应是这份工作最需要的，提问的问题应该多样。"
            boss_memory = memorys['boss'][bid] if bid in memorys['boss'].keys() else None
            geek_dialog, boss_dialog = dialog_play(model,tokenizer,doc_data['resume'][gid],doc_data['JD'][bid],dialog_data[gid+"-"+bid]['dialog_len'],geek_strategy = geek_old_strategy,boss_strategy = boss_old_strategy,geek_memory=memorys['geek'][gid],boss_memory = boss_memory)
            geek_predict = geek_generate(model,tokenizer,boss_dialog,geek_dialog,doc_data['resume'][gid],doc_data['JD'][bid],action='evaluation')
            boss_predict = boss_generate(model,tokenizer,boss_dialog,geek_dialog,doc_data['resume'][gid],doc_data['JD'][bid],action='evaluation')
            
            predicts['geek'][gid+'-'+ bid] = {}
#             predicts['boss'][gid+'-'+ bid] = {}
            
            predicts['geek'][gid+'-'+ bid]['predict'] = geek_predict
            # predicts['boss'][gid+'-'+ bid]['predict'] = boss_predict
            
            predicts['geek'][gid+'-'+ bid]['dialog'] = [geek_dialog, boss_dialog]
            
    print("testing boss")
    for bid in tqdm(test_sample['boss']):
        gids = test_sample['boss'][bid]['pos'] + test_sample['boss'][bid]['neg']
        for gid in gids: #测试的时候就不更新策略
            geek_old_strategy = strategies['geek'][gid] if gid in strategies['geek'].keys() else "回复内容应该和岗位所需技能或者工作经历相关，回复内容应简洁准确，回复内容应突出自己的特点。"
            boss_old_strategy = strategies['boss'][bid] if bid in strategies['boss'].keys() else "提问内容应该和岗位所需技能或者工作经历相关，提问内容应是这份工作最需要的，提问的问题应该多样。"
            geek_memory = memorys['geek'][gid] if bid in memorys['geek'].keys() else None
            geek_dialog, boss_dialog = dialog_play(model,tokenizer,doc_data['resume'][gid],doc_data['JD'][bid],dialog_data[gid+"-"+bid]['dialog_len'],geek_strategy = geek_old_strategy,boss_strategy = boss_old_strategy,boss_memory= memorys['boss'][bid],geek_memory=geek_memory)
            geek_predict = geek_generate(model,tokenizer,boss_dialog,geek_dialog,doc_data['resume'][gid],doc_data['JD'][bid],action='evaluation')
            boss_predict = boss_generate(model,tokenizer,boss_dialog,geek_dialog,doc_data['resume'][gid],doc_data['JD'][bid],action='evaluation')
            
            # predicts['geek'][gid+'-'+ bid] = {}
            predicts['boss'][gid+'-'+ bid] = {}
            
            # predicts['geek'][gid+'-'+ bid]['predict'] = geek_predict
            predicts['boss'][gid+'-'+ bid]['predict'] = boss_predict
            
            predicts['boss'][gid+'-'+ bid]['dialog'] = [geek_dialog, boss_dialog]
            
    return predicts #{'geek':{'gid-bid':{'predict':xx},"dialog":[geek_dialog, boss_dialog]},'boss':{'gid-bid':{'predict':xx},"dialog":[geek_dialog, boss_dialog]}}


def main(args):
    tokenizer = LlamaTokenizer.from_pretrained(args.model_path_or_name, use_fast=False, trust_remote_code=True)
    model = LLM(model=args.model_path_or_name,trust_remote_code=True)
    
    doc_data,train_sample,test_sample=load_split_dataset(args.doc_path,args.sample_path)

    # strategies = train(args,doc_data,train_sample,dialog_data,model,tokenizer)
    
    if args.start != -1:
        def dict_slice(adict, start, end):
            keys = list(adict.keys())
            dict_slice = {}
            for k in keys[start:end]:
                dict_slice[k] = adict[k]
            return dict_slice
        
        test_sample['geek'] = dict_slice(test_sample['geek'],args.start,args.end)
        test_sample['boss'] = dict_slice(test_sample['boss'],args.start,args.end)
    
    with open(args.dialog_path,'r') as t:
        dialog_data = json.load(t)
    
    # we save examples in interview training
    with open("xxx",'r') as t:
        memorys = json.load(t)
        
    print("training end")
    
    if args.start != -1:
        with open(args.output_strategy,'r') as t:
            strategies = json.load(t)

        
        with open(args.ori_predict_path,'r') as t:
            predict_dialog = json.load(t)
        predicts = test_only_predicts(args,doc_data,test_sample,predict_dialog,model,tokenizer)
        # print("testing end")

        with open(args.output,'w') as t:
            json.dump(predicts,t)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--sample_path',
                        type=str,
                        default='',)
    parser.add_argument('--doc_path',
                        type=str,
                        default='')
    parser.add_argument('--dialog_path',
                        type=str,
                        default='')
    parser.add_argument('--ori_predict_path',
                        type=str,
                        default='')
    parser.add_argument('--model_path_or_name',
                        type=str,
                        default=None,)
    
    parser.add_argument("--seed",
                        type=int,
                        default=1234)
    
    parser.add_argument('--output_strategy',
                        type=str,
                        default="xxx")

    parser.add_argument('--output',
                        type=str,
                        default="xxx")
    
    parser.add_argument('--start',
                        type=int,
                        default=-1)
    parser.add_argument('--end',
                        type=int,
                        default=-1)

    args = parser.parse_args()
    main(args)
