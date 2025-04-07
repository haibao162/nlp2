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
        memory = self.slot_filling(memory)
        return memory

    def intent_recognition(self, memory):
        max_score = -1
        for node_name in memory['available_nodes']:
            node_info = self.nodes_info[node_name]
            score = self.get_node_score(memory['query'], node_info)

    def generate_response(self, query, memory):
        memory['query'] = query
        memory = self.nlu(memory)
        memory = self.dst(memory)
        memory = self.dpo(memory)
        memory = self.nlg(memory)
        return memory



if __name__ == '__main__':
    ds = DailogueSystem()
    print(ds.slot_to_qv)
    memory = {"available_nodes":["scenario-买衣服node1"]}

    while True:
        query = input("user: ")
        memory = ds.generate_response(query, memory)
        print("System: ", memory["response"])