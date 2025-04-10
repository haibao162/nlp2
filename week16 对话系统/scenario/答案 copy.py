import json
import pandas
import re

'''
对话系统
基于场景脚本完成多轮对话
'''

class DialogueSystem:
    def __init__(self):
        self.load()

    def load(self):
        self.all_node_info = {} #key = 节点id， value = node info
        self.load_scenario("scenario-买衣服.json")
        self.slot_info = {} #key = slot name, value = slot value
        self.load_template()

        #初始化一个专门的节点用于实现在任意时刻的重听
        self.init_repeat_node()

    def init_memory(self):
        memory = {}
        memory['available_node'] = ["scenario-买衣服-node1"]
        return memory
    
    def load_scenario(self, path):
        scenario_name = path.replace('.json', '')
        with open(path, 'r', encoding='utf-8') as f:
            scenario_data = json.load(f)
        for node_info in scenario_data:
            node_id = node_info['id']
            node_id = scenario_name + '-' + node_id
            if 'childnode' in node_info:
                node_info['childnode'] = [scenario_name + '-' + child for child in node_info['childnode']]

            self.all_node_info[node_id] = node_info

    def nlu(self, memory):
        memory = self.get_intent(memory)
        memory = self.get_slot(memory)
        return memory
    
    def get_intent(self, memory):
        # 获取意图
        hit_node = None
        hit_score = -1
        for node_id in memory['available_node']:
            score = self.get_node_score(node_id, memory)
            if score > hit_score:
                hit_node = node_id
                hit_score = score
        memory['hit_node'] = hit_node
        memory['hit_score'] = hit_score
        return memory

    def get_node_score(self, node_id, memory):
        #计算意图得分
        intent_list = self.all_node_info[node_id]['intent']
        query = memory['query']
        scores = []
        for intent in intent_list:
            score = self.similarity(query, intent)
            scores.append(score)
        return max(scores)
    
    def similarity(self, query, intent):
        # jaccard距离(欧式距离，cos距离)
        intersect = len(set(query) & set(intent))
        union = len(set(query) | set(intent))
        return intersect / union
    
    def get_slot(self, memory):
        # 获取槽位
        hit_node = memory['hit_node']
        for slot in self.all_node_info[hit_node].get('slot', []):
            if slot not in memory:
                values = self.slot_info[slot][1]
                info = re.search(values, memory['query'])
                if info is not None:
                    memory[slot] = info.group()
        return memory

    def dst(self, memory):
        # 对话状态跟踪
        hit_node = memory['hit_node']
        print('hit_node1', hit_node)
        for slot in self.all_node_info[hit_node].get('slot', []):
            if slot not in memory:
                memory['require_slot'] = slot
                return memory
        memory['require_slot'] = None

        print('hit_node2', hit_node)
        if hit_node == 'special_repeat_node': #特殊处理重听
            memory["state"] = "repeat"
        else:
            memory["state"] = None
        return memory
    
    def pm(self, memory):
        # 对话策略执行
        if memory['require_slot'] is not None:
            # 反问策略
            memory['available_node'] = [memory['hit_node']]
            memory['policy'] = 'ask'
        elif memory['state'] == 'repeat':
            #重听策略  不对memory做修改，只更新policy
            memory['policy'] = 'repeat'
        else:
            memory['available_node'] = self.all_node_info[memory['hit_node']].get('childnode', [])
            memory['policy'] = 'answer'
        return memory
    
    def nlg(self, memory):
        # 生成回复
        if memory['policy'] == "ask":
            slot = memory['require_slot']
            reply = self.slot_info[slot][0]
        elif memory['policy'] == "repeat":
            reply = memory['reply']
        else:
            reply = self.all_node_info[memory['hit_node']]['response']
            reply = self.replace_template(reply, memory)
        memory['reply'] = reply
        return memory
    
    def replace_template(self, reply, memory):
        hit_node = memory['hit_node']
        for slot in self.all_node_info[hit_node].get('slot', []):
            reply = re.sub(slot, memory[slot], reply)
        return reply


    
    def run(self, query, memory):
        if memory == {}:
            memory = self.init_memory()
        memory['query'] = query
        memory = self.nlu(memory)
        memory = self.dst(memory) # 对话状态跟踪
        memory = self.pm(memory) # 策略管理
        memory = self.nlg(memory) # 自然语言生成
        return memory

    def load_template(self):
        df = pandas.read_excel("./slot_fitting_templet.xlsx")
        for i in range(len(df)):
            slot = df['slot'][i]
            query = df['query'][i]
            values = df['values'][i]
            self.slot_info[slot] = [query, values]

    def init_repeat_node(self):
        node_id = 'special_repeat_node'
        node_info = {'id': node_id, 'intent': ["你说啥", "再说一遍"]}
        self.all_node_info[node_id] = node_info #记录这个新节点
        for node_info in self.all_node_info.values():
            node_info['childnode'] = node_info.get('childnode', []) + [node_id] #将这个新节点作为所有节点的子节点


if __name__ == '__main__':
    ds = DialogueSystem()
    print(ds.slot_info)
    memory = {}
    while True:
        query = input("user: ")
        memory = ds.run(query, memory)
        print(memory['reply'])
        print()
