# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Adapted from https://github.com/MARIO-Math-Reasoning/Super_MARIO
from __future__ import annotations
from termcolor import colored
from typing import Dict, Any, Optional, Type, List, Tuple, Callable, Union
from pydantic import field_validator
from vllm.outputs import CompletionOutput, RequestOutput
from rstar_deepthink.agents.utils import math_equiv as is_equiv
from rstar_deepthink.nodes.base_node import BaseNode
from rstar_deepthink.nodes import MCTSNode
from rstar_deepthink.constants import (
    TOO_MANY_CODE_ERRORS, 
    TOO_MANY_STEPS, 
    NO_VALID_CHILD, 
    CODE_END,
)
from .tree import BaseTree, code_execution
from .beam_search import BS

is_disable = False

def pwarn(text, disable=False):
    if not disable:
        print(text)

def pinfo(text, disable=False):
    if not disable:
        print(text)

class MCTS(BS):
    search_node: Type[BaseNode] = None

    intermediate_metric: Dict = {
        "question": "",
        "gt": "", 
        "answers": [],
        "judgements": [],
        "value_estimate": [],
        "rollout_indexs": [],
    }

    @field_validator("config")
    def validate_config(cls, cfg: Any):
        BaseTree.validate_config(cfg)
        if not cfg.mode == "mcts":
            raise ValueError(f"Wrong value for config mode.")
        if cfg.stop is None:
            raise ValueError(f"Wrong value for config stop, cannot be None")
        return cfg

    def create_node(self, parent: Optional[Type[MCTSNode]] = None) -> Type[MCTSNode]:
        return MCTSNode(
            parent=parent, 
            additional_state_keys=self.NODE_KEYS,
            c_puct=self.config.c_puct,
        )


    def selection(self, from_root=False) -> Optional[Type[MCTSNode]]:
        if from_root:
            start_node = self.root
        else:
            start_node = self.search_node   # the search node is the current_nodes[0] 
                                            #     specified in select_next_step function
        # select a child node
        node = start_node
        if node is None: return None

        # from the start_node, explore the non-terminal children with UCT 
        # notes: 
        #    this step is different from the Mario code, 
        #       where the selection process always starts from the root.
        if node.has_children() or node.is_terminal: # an potentiall error here "or node.is_terminal" may not be correct, 
                                                    #    should be "and not node.is_terminal"?
            next_node = self.select_child(node)     # To encourage exploration, select from non-terminal children
            if next_node is None:                   # if None，it mean all children are terminal
                node.is_terminal = True             #    if all children are terminal then the node is terminal??
                                                    # Does it miss a "break" here?
            node = next_node

        # The selected node is None if 
        #   (i) it is a terminal node
        #   or (ii) all of its childrens are terminal nodes
        # The selected node is not None if 
        #   it is a non-terminal leaf node that have no children 
        return None if (node is None or node.is_terminal) else node

    def select_child(self, node: Type[MCTSNode]) -> Optional[Type[MCTSNode]]:
        best_value = -float("inf")
        best_childs = []

        for child in node.children:
            if child.is_terminal:
                continue
            puct_value = child.puct()
            if puct_value == best_value:
                best_childs.append(child)
            elif puct_value > best_value:
                best_value = puct_value
                best_childs = [child]

        #return random.choice(best_childs) if best_childs else None
        return best_childs[0] if best_childs else None

    def expand_node(self, outputs: List[CompletionOutput], node: Type[MCTSNode]) -> None:
        for idx, output in enumerate(outputs):
            if not output.stop_reason: output.stop_reason = ""
            step_result, parser_result = self.step_unwrap(output.text + output.stop_reason)
            self.create_child(step_result, parser_result, node, idx)

    def create_child(
        self, 
        step_result: str, 
        parser_result: Dict[str, str], 
        node: Type[MCTSNode],
        idx: int,
    ) -> None:
        new_node = self.create_node(parent=node)
        parent_child_count = len(node.children)
        new_node.tag = f"{node.tag}.{parent_child_count + 1}"
        new_node.depth = node.depth + 1

        if parser_result is None:
            new_node.is_terminal = True
            new_node.state["text"] = step_result
            new_node.state["final_answer"] = NO_VALID_CHILD
            self.eval_final_answer(new_node)
        elif parser_result["final_answer"]:
            new_node.is_terminal = True
            new_node.state["text"] = step_result
            new_node.state["final_answer"] = parser_result["final_answer"]
            self.eval_final_answer(new_node)
        elif parser_result["action"]:
            observation = code_execution(node, parser_result)
            new_node.state["action"] = parser_result["action"]
            new_node.state["action_input"] = parser_result["action_input"]
            new_node.state["observation"] = observation
            if CODE_END in parser_result["action_input"]:
                observation = self.obs_wrap(observation)
                new_node.state["text"] = f"{step_result}{self.config.step_delim}{observation}"
            else:
                new_node.state["text"] = step_result
                
            if "error" in observation.lower():
                new_node.consecutive_errors = node.consecutive_errors + 1
                if new_node.consecutive_errors >= self.config.errors_threshold:
                    observation = self.obs_wrap(observation)
                    step_result = step_result + CODE_END if CODE_END not in step_result else step_result
                    new_node.state["text"] = f"{step_result}{self.config.step_delim}{observation}"
                    new_node.is_terminal = True
                    new_node.state["final_answer"] = TOO_MANY_CODE_ERRORS
                    self.eval_final_answer(new_node)
        else:
            new_node.state["text"] = step_result

        if not new_node.is_terminal and new_node.depth > self.config.max_depth:
            new_node.is_terminal = True
            new_node.state["final_answer"] = TOO_MANY_STEPS
            self.eval_final_answer(new_node)

        node.children.append(new_node)

    def eval_final_answer(self, node: Type[MCTSNode]) -> None:
        if node.state["final_answer"] in [NO_VALID_CHILD, TOO_MANY_STEPS, TOO_MANY_CODE_ERRORS]:
            # if the final answer is not valid, update the node with negative reward
            node.update(self.config.negative_reward)
            return 
        
        if self.config.is_sampling:
            final_answer = node.state["final_answer"]
            correct = is_equiv(self.ground_truth, final_answer)
            node.update_recursive(self.config.positive_reward if correct else self.config.negative_reward, self.root)
        else:
            # just append the node to candidate_nodes, will update the value in select_next_step()
            self.candidate_nodes.append(node)


    def record_intermediate_metric(self, answer, value_estimate):
        self.intermediate_metric["question"] = self.question
        self.intermediate_metric["gt"] = self.ground_truth
        # each rollout retains the answer with the highest value_estimate
        # Check if the rollout's answer is already in the list
        if self.intermediate_metric["rollout_indexs"] and self.rollout_idx in self.intermediate_metric["rollout_indexs"]:
            # Find the index of the existing rollout
            index = self.intermediate_metric["rollout_indexs"].index(self.rollout_idx)
            if value_estimate > self.intermediate_metric["value_estimate"][index]:
                self.intermediate_metric["answers"][index] = answer
                self.intermediate_metric["judgements"][index] = is_equiv(self.ground_truth, answer)
                self.intermediate_metric["value_estimate"][index] = value_estimate
        else:
            # If the rollout's answer is not in the list, add it
            self.intermediate_metric["answers"].append(answer)
            self.intermediate_metric["judgements"].append(is_equiv(self.ground_truth, answer))
            self.intermediate_metric["value_estimate"].append(value_estimate)
            self.intermediate_metric["rollout_indexs"].append(self.rollout_idx)


    def select_next_step(self, outputs=None, from_root=False) -> None:
        '''
        Update and backup node values from the self.candidate_nodes 
        Then, select a next node for expansion 
        '''
        pinfo(f"\n-> select_next_step")
        # if self.current_nodes is None (i.e., cannot select next node for exploration), the current rollout stops 
        # What’s a better way to describe the cases where this happens?
        self.search_node = self.current_nodes[0] if self.current_nodes else None
        pwarn(self.search_node)
        self.current_nodes = []

        if outputs:        # This is False for expanded_agends, which skips backup operation, go to selection operation  
            pwarn(f"outputs.yes")
            # only update/backup from the self.candidate_nodes
            for candidate_node, output in zip(self.candidate_nodes, outputs): # this must match since we use PRM to scores candidate_nodes
                pwarn(candidate_node)
                pwarn(output.value_estimate)
                pwarn(candidate_node.is_terminal)
                
                if candidate_node.is_terminal and self.config.is_sampling: # what is self.config.is_sampling for?
                    continue
                
                value_estimate = output.value_estimate if output.value_estimate is not None else self.config.negative_reward
                if output.value_estimate is None:
                    candidate_node.is_terminal = True

                # backup
                # TOO_MANY_CODE_ERRORS = "Too many consecutive steps have code errors."
                # TOO_MANY_STEPS = "Fail to sove the problem within limited steps."
                # NO_VALID_CHILD = "Fail to generate parsable text for next step."
                # where is candidate_node.state["final_answer"] computed? 
                if candidate_node.is_terminal and candidate_node.state["final_answer"]:
                    pwarn("is_terminal.yes")
                    # for terminal node: update_recursive
                    if candidate_node.state["final_answer"] in [NO_VALID_CHILD, TOO_MANY_STEPS, TOO_MANY_CODE_ERRORS]:
                        pwarn("invalid_final_answer.yes")
                        candidate_node.update(self.config.negative_reward)
                    else:
                        # save intermediate metric. what does this step actually do?
                        self.record_intermediate_metric(answer=candidate_node.state["final_answer"], value_estimate=value_estimate)

                        candidate_node.update_recursive(value_estimate, self.root)
                else:
                    # for intermediate node: just update the value
                    if self.config.terminal_sample: # Always False
                        pass
                    else:
                        candidate_node.update(value_estimate)

                if self.__class__.is_valid_final_answer_node(candidate_node):
                    self.final_answer_nodes.append(candidate_node)
                    
        selection_node = self.selection(from_root=from_root)
        pwarn(selection_node)

        # The selected node is None if 
        #   (i) it is a terminal node
        #   or (ii) all of its childrens are terminal nodes
        # The selected node is not None if 
        #   it is a non-terminal leaf node that have no children 
        # If the selected node is None, there is no further exploration, the current rollout stops 
        if selection_node is not None:
            self.current_nodes.append(selection_node)
    
    
    def generate_next_step(self, outputs: List[RequestOutput]) -> None:
        self.candidate_nodes = []
        pinfo(f"\n-> generate_next_step")
        cnt = -1
        for current_node, output in zip(self.current_nodes, outputs):
            cnt += 1
            pwarn(cnt)
            pwarn(current_node)
            pwarn(output)
            pwarn(output.value_estimate)
            value_estimate = output.value_estimate
            if value_estimate is not None:  
                self.expand_node(output.outputs, current_node)   # create child nodes from outputs
            else:   # when does it happen? 
                print("generate_next_step value_estimate is None")
                print(current_node)
                value_estimate = self.config.negative_reward
                current_node.is_terminal = True

            # update leaf nodes
            # make self.candidate_nodes
            #     only add generated nodes to self.candidate_nodes if they haven't been visited before 
            #     (i.e., their visiation counts are 0)
            if self.config.update_leaf_value:           # always True 
                # if need update leaf node value, just append the node to candidate_nodes, will update the value in select_next_step()
                for value_node in current_node.children:
                    if value_node not in self.candidate_nodes and value_node.visit_count() < 1:
                        self.candidate_nodes.append(value_node) 
            pwarn(self.candidate_nodes)
                    
                    
    def return_states(self) -> Dict[str, Union[Any, Dict[str, str]]]:
        candidates = [self.root]
        states = {}
        while candidates:
            node = candidates.pop(0)
            states[node.tag] = node.state
            states[node.tag]["value"] = node.value
            states[node.tag]["q_value"] = node.q_value()
            states[node.tag]["visit_count"] = node.visit_count()
            if node.has_children():
                candidates.extend(node.children)
        return states
