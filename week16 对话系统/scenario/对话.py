import json
import pandas as pd
import re

# slot	query	values
#服装类型#	您想买长袖、短袖还是半截袖	长袖|短袖|半截袖
#服装颜色#	您喜欢什么颜色	红|橙|黄|绿|青|蓝|紫
#服装尺寸#	您想要多尺寸	s|m|l|xl|xll
#分期付款期数#	您想分多少期，可以有3期，6期，9期，12期	3|6|9|12
#支付方式#	您想使用什么支付方式	信用卡|支付宝|微信

class DailogueSystem:
    def __init__(self):
        self.load()

    def load(self):
        self.nodes_info = {}
        self.load_scenario('scenario-买衣服.json')
        self.load_slot_template('slot_fitting_templet.xlsx')
        # {'#服装类型#': ('您想买长袖、短袖还是半截袖', '长袖|短袖|半截袖'), 
        #  '#服装颜色#': ('您喜欢什么颜色', '红|橙|黄|绿|青|蓝|紫'), 
        #  '#服装尺寸#': ('您想要多尺寸', 's|m|l|xl|xll'),
        #  '#分期付款期数#': ('您想分多少期，可以有3期，6期，9期，12期', '3|6|9|12'),
        #  '#支付方式#': ('您想使用什么支付方式', '信用卡|支付宝|微信')}

    def load_scenario(self, scenario_file):
        with open(scenario_file, 'r', encoding='utf-8') as f:
            self.scenario = json.load(f)
        scenario_name = scenario_file.split('.')[0] # scenario-买衣服
        for node in self.scenario:
            self.nodes_info[scenario_name + node["id"]] = node
            if "childnode" in node:
                node["childnode"] = [scenario_name + childnode for childnode in node["childnode"]]


    def load_slot_template(self, slot_template_file):
        self.slot_template = pd.read_excel(slot_template_file)
        #slot	query	values
        self.slot_to_qv = {}
        for i, row in self.slot_template.iterrows():
            slot = row["slot"]
            query = row["query"]
            values = row["values"]
            self.slot_to_qv[slot] = (query, values)


    def nlu(self, memory):
        memory = self.intent_recognition(memory)
        # print(memory, 'memory')
        # {'available_nodes': ['scenario-买衣服node1'], 'query': 'a', 'hit_node': 'scenario-买衣服node1'}
        memory = self.slot_filling(memory)
        return memory

    def intent_recognition(self, memory):
        max_score = -1
        for node_name in memory['available_nodes']:
            node_info = self.nodes_info[node_name]
            score = self.get_node_score(memory['query'], node_info)
            if score > max_score:
                max_score = score
                memory['hit_node'] = node_name
        return memory

    def get_node_score(self, query, node_info):
        # 跟node中的intent算分
        intent_list = node_info["intent"]
        score = 0
        for intent in intent_list:
            score = max(score, self.sentence_match_score(query, intent))
        return score

    def sentence_match_score(self, string1, string2):
        # 计算两个句子之间的相似度,使用jaccard距离
        s1 = set(string1)
        s2 = set(string2)
        return len(s1.intersection(s2)) / len(s1.union(s2))
    
    def slot_filling(self, memory):
        # 槽位填充模块，根据当前节点中的slot，对query进行槽位填充
        # 根据命中的节点，获取对应的slot
        slot_list = self.nodes_info[memory['hit_node']].get('slot', [])
        # 对query进行槽位填充
        # "slot":["#服装类型#", "#服装颜色#", "#服装尺寸#"],
        for slot in slot_list:
            slot_values = self.slot_to_qv[slot][1]
            # 长袖|短袖|半截袖
            if re.search(slot_values, memory["query"]):
                memory[slot] = re.search(slot_values, memory["query"]).group()
        return memory
    
    def dst(self, memory):
        #确认当前hit_node所需要的所有槽位是否已经齐全，'#服装类型#': '长袖'
        # 一开始slot不在memory中，require_slot为服装类型
        slot_list = self.nodes_info[memory['hit_node']].get('slot', [])
        for slot in slot_list:
            if slot not in memory:
                memory['require_slot'] = slot
                return memory
        memory["require_slot"] = None
        return memory
    
    def dpo(self, memory):
        #如果require_slot为空，则执行当前节点的操作,否则进行反问
        if memory["require_slot"] is None:
            memory["policy"] = "reply"
            childnodes = self.nodes_info[memory['hit_node']].get('childnode', [])
            memory["avaliable_nodes"] = childnodes
            # 执行动作 take action
        else:
            memory["policy"] = "ask"
            memory["available_nodes"] = [memory['hit_node']] #停留在当前节点，直到槽位填满
        return memory
        


    def generate_response(self, query, memory):
        memory['query'] = query
        memory = self.nlu(memory)
        memory = self.dst(memory)
        # print(memory, 'dst')
        # {'available_nodes': ['scenario-买衣服node1'], 
        #  'query': '长袖', 'hit_node': 'scenario-买衣服node1', 
        #  'require_slot': '#服装颜色#', '#服装类型#': '长袖'}
        # memory = self.dpo(memory)
        # memory = self.nlg(memory)
        return memory



if __name__ == '__main__':
    ds = DailogueSystem()
    print(ds.slot_to_qv)
    memory = {"available_nodes":["scenario-买衣服node1"]}

    while True:
        query = input("user: ")
        memory = ds.generate_response(query, memory)
        # print("System: ", memory["response"])