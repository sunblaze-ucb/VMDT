from typing import List, Dict, Tuple, Optional
import random
import argparse
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from adversarial.common.utils import read_jsonl, write_jsonl, set_seed
from adversarial.t2v.t2v_utils import T2VInstance
from adversarial.t2v.models.mochi import Mochi
from adversarial.t2v.models.cogvideox import CogVideoX


class GCGAttack:
    def __init__(
        self,
        model,
        num_adversarial_tokens: int,
        num_return_prompts: int,
        iterations: int,
        k: int,
        forbidden_num: int,
        weight_target: float,
        weight_clean: float,
        device: str, 
        dtype: torch.dtype,
    ):  
        self.model = model
        self.text_encoder = model.pipe.text_encoder
        self.tokenizer = model.pipe.tokenizer
        self.num_adversarial_tokens = num_adversarial_tokens
        self.num_return_prompts = num_return_prompts
        self.iterations = iterations
        self.k = k
        self.forbidden_num = forbidden_num
        self.weight_target = weight_target
        self.weight_clean = weight_clean
        self.device = device
        self.dtype = eval(dtype)
        
        special_ids = [
            self.tokenizer.bos_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.pad_token_id,
            self.tokenizer.unk_token_id,
            self.tokenizer.mask_token_id,
        ]
        self.special_ids = list(set([idx for idx in special_ids if idx is not None]))
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.embed_weights = self.text_encoder.get_input_embeddings().weight.to(self.dtype)
    
    def tokenize_and_get_optimize_slice(self, ori_text, tar_text, adv_symbol=".", adv_string_length=5):
        tokenized_ori_text = self.tokenizer(text=ori_text, return_tensors="pt").input_ids
        tokenized_tar_text = self.tokenizer(text=tar_text, return_tensors="pt").input_ids

        # Prepare the adversarial text by appending the adversarial symbol string (before eos token)
        adv_token = self.tokenizer.encode(adv_symbol, add_special_tokens=False)[0]
        adv_tokens = torch.full((1, adv_string_length), adv_token).long()
        tokenized_adv_text = torch.cat((tokenized_ori_text[:, :-1], adv_tokens, tokenized_ori_text[:, -1:]), dim=1)

        # Calculate the start and end position of the adversarial content in the tokenized representation
        start = tokenized_ori_text.size(1) - 1
        end = start + adv_string_length

        return tokenized_ori_text, tokenized_tar_text, tokenized_adv_text, [start, end]
    
    def is_ascii_token(self, tokenizer):
        ascii_tokens = []
        for i in range(len(tokenizer.get_vocab())):
            token = tokenizer.decode([i])
            if token.isascii() and token.isprintable():
                ascii_tokens.append(i)
        return ascii_tokens
    
    def filter_similar_tokens(self, target_token, num_tokens=20):
        # Calculate cosine similarity between target token embedding and all other token embeddings
        target_emb = self.embed_weights[target_token].unsqueeze(0)
        cos_sims = F.cosine_similarity(self.embed_weights, target_emb, dim=1)
        # Get the indices of the top `num_tokens` most similar tokens
        _, top_tokens = torch.topk(cos_sims, num_tokens)
        return top_tokens.tolist()
    
    def token_gradients_targeted(self, tokenized_adv_text, ori_emb_text, tar_emb_text, input_start_end):
        input_start, input_end = input_start_end
        input_ids = tokenized_adv_text.squeeze(0)

        one_hot = torch.zeros(
            input_end - input_start,
            self.embed_weights.shape[0],
            device=self.device,
            dtype=self.dtype
        )
        one_hot.scatter_(
            1,
            input_ids[input_start: input_end].unsqueeze(1),
            torch.ones(one_hot.shape[0], 1, device=self.device, dtype=self.dtype)
        )
        one_hot.requires_grad_()
        input_embeds = (one_hot @ self.embed_weights).unsqueeze(0)

        # now stitch it together with the rest of the embeddings
        embeds = self.text_encoder.get_input_embeddings()(input_ids.unsqueeze(0)).detach()
        full_embeds = torch.cat(
            [
                embeds[:, :input_start, :],
                input_embeds,
                embeds[:, input_end:, :]
            ],
        dim=1)
        
        model_predictions = self.text_encoder(inputs_embeds=full_embeds).last_hidden_state.mean(dim=1).view(-1)
        
        loss_value = self.weight_target * (1 - self.cos(model_predictions, tar_emb_text)) + self.weight_clean * (self.cos(model_predictions, ori_emb_text))
        loss_value.backward()

        grad = one_hot.grad.clone()
        grad = grad / grad.norm(dim=-1, keepdim=True)

        return grad, loss_value
    
    def substitute_topk(self, gradient, current_ids, k=3, not_allowed_tokens=None):
        # gradient: adv_string_length * vocab_size
        top_grad, top_grad_token_id = [], []
        for i in range(len(gradient)):
            gradient[i][self.special_ids] = 1e10
            gradient[i][not_allowed_tokens] = 1e10
            idx = torch.argmax(-gradient[i], keepdim=False).item()
            top_grad.append(-gradient[i][idx].cpu().item())
            top_grad_token_id.append(idx)
        top_grad_idx_list = np.argsort(np.array(top_grad))[-k:]
        for top_grad_idx in top_grad_idx_list:
            current_ids[top_grad_idx] = top_grad_token_id[top_grad_idx]
        return current_ids
    
    def transform(self, clean_prompt, adv_target_property, adv_target_prompt) -> List[str]:
    
        tokenized_ori_text, tokenized_tar_text, tokenized_adv_text, [start, end] = self.tokenize_and_get_optimize_slice(clean_prompt, adv_target_prompt, adv_string_length=self.num_adversarial_tokens)
        tokenized_ori_text, tokenized_tar_text, tokenized_adv_text = tokenized_ori_text.to(self.device), tokenized_tar_text.to(self.device), tokenized_adv_text.to(self.device)

        text_list = []
        loss_list = []

        allowed_tokens = self.is_ascii_token(self.tokenizer)

        target_token_id = self.tokenizer.encode(adv_target_property, add_special_tokens=False)[0]
        forbidden_tokens = self.filter_similar_tokens(target_token_id, num_tokens=self.forbidden_num)
        allowed_tokens = [tok for tok in allowed_tokens if tok not in forbidden_tokens]
        not_allowed_tokens = [tok for tok in range(self.embed_weights.shape[0]) if tok not in allowed_tokens]
        
        with torch.no_grad():
            ori_emb_text = self.text_encoder(input_ids=tokenized_ori_text).last_hidden_state.mean(dim=1).view(-1)
            tar_emb_text = self.text_encoder(input_ids=tokenized_tar_text).last_hidden_state.mean(dim=1).view(-1)

        for _ in range(self.iterations):
            current_adv_text = self.tokenizer.batch_decode(tokenized_adv_text, skip_special_tokens=True)[0]
            text_list.append(current_adv_text)

            grad, loss = self.token_gradients_targeted(tokenized_adv_text, ori_emb_text, tar_emb_text, [start, end])
            loss_list.append(loss.item())
            tokenized_adv_text[0, start:end] = self.substitute_topk(grad, tokenized_adv_text[0, start:end], k=self.k, not_allowed_tokens=not_allowed_tokens)

        sorted_text_and_loss = sorted(zip(text_list, loss_list), key=lambda x: x[1])
        sorted_text_list = [item[0] for item in sorted_text_and_loss]
        return sorted_text_list[:self.num_return_prompts]
    

class GreedyAttack:
    def __init__(
        self, 
        model, 
        num_adversarial_chars: int, 
        num_return_prompts: int,
        iterations: int, 
        device: str, 
        dtype: torch.dtype,
    ): 
        self.model = model
        self.text_encoder = model.pipe.text_encoder
        self.tokenizer = model.pipe.tokenizer
        self.num_adversarial_chars = num_adversarial_chars
        self.num_return_prompts = num_return_prompts
        self.iterations = iterations
        self.device = device
        self.dtype = eval(dtype)
        
        self.char_table = self.get_char_table()
        self.max_length = model.max_length
        self.cos_fn = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        
    def get_char_table(self):
        char_table=['·','~','!','@','#','$','%','^','&','*','(',')','=','-','*','+','.','<','>','?',',','\'',';',':','|','\\','/']
        for i in range(ord('A'),ord('Z')+1):
            char_table.append(chr(i))
        for i in range(0,10):
            char_table.append(str(i))
        return char_table
    
    def get_text_embeds(self, prompt: str) -> torch.Tensor:
        input_ids = self.tokenizer(
            prompt, padding='max_length', max_length=self.model.max_length,
            truncation=True, return_tensors='pt'
        ).input_ids.to(self.device)
        with torch.no_grad():
            embeds = self.text_encoder(input_ids).last_hidden_state.squeeze(0)
        return embeds

    def get_mask(self, sentence_list: List[str], object_word: str, thres: float = 0.9) -> torch.Tensor:
        embedding_differences = [
            self.get_text_embeds(sentence.replace(object_word, "")) - self.get_text_embeds(sentence)
            for sentence in sentence_list
        ]
        aggregated_signs = torch.stack(embedding_differences).sign().sum(dim=0)
        mask = (aggregated_signs.abs() > thres*len(sentence_list)).to(torch.int)
        return mask

    def greedy_search(self, prompt: str, mask: torch.Tensor) -> str:
        prompt_embeds = self.get_text_embeds(prompt)
        adv_prompt = prompt + " "
        self.candidates = {}
        for i in range(self.num_adversarial_chars):
            adv_prompt += " "
            adv_prompt = self.greedy_search_helper(prompt_embeds, adv_prompt, -1, None)
        for i in range(self.iterations):
            for k in range(self.num_adversarial_chars, 0, -1):
                adv_prompt = self.greedy_search_helper(prompt_embeds, adv_prompt, -k, mask)
        adv_prompts = sorted(self.candidates.items(), key=lambda x: x[1])[:self.num_return_prompts]
        adv_prompts = [x[0] for x in adv_prompts]
        return adv_prompts
    
    def greedy_search_helper(self, prompt_embeds: torch.Tensor, adv_prompt: str, pos: int, mask: torch.Tensor = None):
        min_cos = 1.0
        best_prompt = adv_prompt
        for c in self.char_table:
            modified_prompt = list(adv_prompt)
            modified_prompt[pos] = c
            modified_prompt = "".join(modified_prompt)
            if modified_prompt in self.candidates:
                continue
            modified_embeds = self.get_text_embeds(modified_prompt)
            if mask is None:
                cos = self.cos_fn(prompt_embeds.view(-1), modified_embeds.view(-1))
            else:
                cos = self.cos_fn(prompt_embeds.view(-1) * mask.view(-1), modified_embeds.view(-1) * mask.view(-1))
            self.candidates[modified_prompt] = cos.item()
            if cos < min_cos:
                min_cos = cos
                best_prompt = modified_prompt
        return best_prompt

    def transform(self, prompt: str, sentence_list: List[str], target_property: str) -> List[str]:
        mask = self.get_mask(sentence_list=sentence_list, object_word=target_property)
        adv_prompts = self.greedy_search(prompt, mask)
        return adv_prompts
    

class GeneticAttack:
    def __init__(
        self, 
        model, 
        num_adversarial_chars: int, 
        num_return_prompts: int,
        iterations: int, 
        init_population_size: int,
        survivor_count: float,
        mutation_rate: float,
        device: str, 
        dtype: torch.dtype,
    ):
        
        self.model = model
        self.text_encoder = model.pipe.text_encoder
        self.tokenizer = model.pipe.tokenizer
        self.num_adversarial_chars = num_adversarial_chars
        self.num_return_prompts = num_return_prompts
        self.iterations = iterations
        self.init_population_size = init_population_size
        self.survivor_count = survivor_count
        self.mutation_rate = mutation_rate
        self.device = device
        self.dtype = eval(dtype)
        
        self.char_table = self.get_char_table()
        self.max_length = model.max_length
        self.cos_fn = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        
    def get_char_table(self):
        char_table=['·','~','!','@','#','$','%','^','&','*','(',')','=','-','*','+','.','<','>','?',',','\'',';',':','|','\\','/']
        for i in range(ord('A'),ord('Z')+1):
            char_table.append(chr(i))
        for i in range(0,10):
            char_table.append(str(i))
        return char_table
    
    def get_text_embeds(self, prompt: str) -> torch.Tensor:
        input_ids = self.tokenizer(
            prompt, padding='max_length', max_length=self.model.max_length,
            truncation=True, return_tensors='pt'
        ).input_ids.to(self.device)
        with torch.no_grad():
            embeds = self.text_encoder(input_ids).last_hidden_state.squeeze(0)
        return embeds

    def get_mask(self, sentence_list: List[str], object_word: str, thres: float = 0.9) -> torch.Tensor:
        embedding_differences = [
            self.get_text_embeds(sentence.replace(object_word, "")) - self.get_text_embeds(sentence)
            for sentence in sentence_list
        ]
        aggregated_signs = torch.stack(embedding_differences).sign().sum(dim=0)
        mask = (aggregated_signs.abs() > thres*len(sentence_list)).to(torch.int)
        return mask
    
    def initialize_population(self) -> List[str]:
        population = []
        for _ in range(self.init_population_size):
            prompt = "".join(random.choices(self.char_table, k=self.num_adversarial_chars))
            population.append(prompt)
        return population

    def crossover(self, parent1: str, parent2: str) -> Tuple[str, str]:    
        crossover_point = random.randint(1, len(parent1)-1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2
    
    def mutate(self, child1: str, child2: str) -> Tuple[str, str]:
        mutation_idx = random.randint(0, len(child1)-1)
        mutation_char = random.choice(self.char_table)
        if random.random() < self.mutation_rate:
            child1 = child1[:mutation_idx] + mutation_char + child1[mutation_idx + 1:]
        if random.random() < self.mutation_rate:
            child2 = child2[:mutation_idx] + mutation_char + child2[mutation_idx + 1:]
        return child1, child2
    
    def fitness(self, prompt: str, adv_chars: str, prompt_embeds: torch.Tensor, mask: torch.Tensor = None) -> float:
        if adv_chars in self.cache:
            return self.cache[adv_chars]
        adv_prompt = f"{prompt} {adv_chars}"
        adv_embeds = self.get_text_embeds(adv_prompt)
        if mask is None:
            score = self.cos_fn(prompt_embeds.view(-1), adv_embeds.view(-1)).item()
        else:
            score = self.cos_fn(prompt_embeds.view(-1) * mask.view(-1), adv_embeds.view(-1) * mask.view(-1)).item()
        self.cache[adv_chars] = score
        return score

    def genetic_search(self, prompt: str, mask: torch.Tensor) -> str:
        prompt_embeds = self.get_text_embeds(prompt)
        population = self.initialize_population()
        self.cache: Dict[str, float] = {}
        for _ in range(self.iterations):
            offspring = []
            for parent1 in population:
                parent2 = random.choice(population)
                child1, child2 = self.crossover(parent1, parent2)
                child1, child2 = self.mutate(child1, child2)
                offspring.extend([child1, child2])
            population.extend(offspring)
            population = list(set(population))
            fitness_scores = [
                self.fitness(prompt, adv_chars, prompt_embeds, mask)
                for adv_chars in population
            ]
            sorted_indices = np.argsort(fitness_scores)
            population = [population[idx] for idx in sorted_indices[:self.survivor_count]]
        return [f"{prompt} {adv_chars}" for adv_chars in population[:self.num_return_prompts]]
    
    def transform(self, prompt: str, sentence_list: List[str], target_property: str) -> List[str]:
        mask = self.get_mask(sentence_list=sentence_list, object_word=target_property)
        adv_prompts = self.genetic_search(prompt, mask)
        return adv_prompts


def load_model(model_name: str, device: str, dtype: torch.dtype) -> nn.Module:
    if model_name == "CogVideoX-2b":
        model = CogVideoX(model_size="2b", compile=False, device=device, dtype=dtype)
    elif model_name == "CogVideoX-5b":
        model = CogVideoX(model_size="5b", compile=False, device=device, dtype=dtype)
    elif model_name == "mochi-1-preview":
        model = Mochi(compile=False, device=device, dtype=dtype)
    else:
        raise ValueError("Invalid model")
    model._del_transformer()
    model._del_vae()
    return model

def init_attack(model, args):
    if args.attack == "gcg":
        attack = GCGAttack(
            model=model,
            num_adversarial_tokens=args.num_adversarial_tokens,
            num_return_prompts=args.num_return_prompts,
            iterations=args.iterations,
            k=args.k,
            forbidden_num=args.forbidden_num,
            weight_target=args.weight_target,
            weight_clean=args.weight_clean,
            device=args.device,
            dtype=args.dtype,
        )
    elif args.attack == "greedy":
        attack = GreedyAttack(
            model=model,
            num_adversarial_chars=args.num_adversarial_chars,
            num_return_prompts=args.num_return_prompts,
            iterations=args.iterations,
            device=args.device,
            dtype=args.dtype,
        )
    elif args.attack == "genetic":
        attack = GeneticAttack(
            model=model,
            num_adversarial_chars=args.num_adversarial_chars,
            num_return_prompts=args.num_return_prompts,
            iterations=args.iterations,
            init_population_size=args.init_population_size,
            survivor_count=args.survivor_count,
            mutation_rate=args.mutation_rate,
            device=args.device,
            dtype=args.dtype,
        )
    else:
        raise ValueError("Invalid attack")
    return attack

def prepare_attack_call_args(instance, attack):
    if attack == "gcg":
        return {
            "clean_prompt": instance.clean_prompt,
            "adv_target_property": instance.target_property._get_core_property(),
            "adv_target_prompt": instance.target_prompt,
        }
    elif attack in ["greedy", "genetic"]:
        sentences = [
            "A black panther lying in a jungle",
            "A fishing boat on a lake at sunrise",
            "A tea cup on a saucer with a teapot",
            "A man playing guitar on a street corner",
            "A group of flamingos standing in a pond",
            "A fireflies in a field at dusk",
            "A train chugging through a countryside",
            "A butterfly on a colorful flower",
            "A soccer game being played on a stadium",
            "A man kayaking down a river through rapids",
        ]
        clean_property = instance.target_property._get_core_property()
        mask_sentences = [f"{sentence} and a {clean_property}" for sentence in sentences]
        return {
            "prompt": instance.clean_prompt,
            "sentence_list": mask_sentences,
            "target_property": clean_property,
        }
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # core args
    parser.add_argument("--attack", type=str, required=True) # gcg, greedy, genetic
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--max_prompts", type=int, default=10000)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    
    # model args
    parser.add_argument("--surrogate_model", type=str, required=True) # CogVideoX-2b, CogVideoX-5b, mochi-1-preview
    
    # shared attack args
    parser.add_argument("--num_return_prompts", type=int, default=1)
    parser.add_argument("--iterations", type=int, required=True) # gcg: 400, greedy: 4, genetic: 50
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="torch.bfloat16")
    
    # gcg attack args
    parser.add_argument("--num_adversarial_tokens", type=int, default=5)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--forbidden_num", type=int, default=20)
    parser.add_argument("--weight_target", type=float, default=1.0)
    parser.add_argument("--weight_clean", type=float, default=1.0)
    
    # greedy/genetic attack args
    parser.add_argument("--num_adversarial_chars", type=int, default=5)
    parser.add_argument("--init_population_size", type=int, default=10) # genetic only
    parser.add_argument("--survivor_count", type=int, default=20) # genetic only
    parser.add_argument("--mutation_rate", type=float, default=0.3) # genetic only
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    instances = [T2VInstance.parse_obj(x) for x in read_jsonl(args.input_file)][:args.max_prompts]
    model = load_model(args.surrogate_model, args.device, args.dtype)
    attack = init_attack(model, args)
    
    new_instances = []
    for instance in tqdm(instances):
        new_instance = copy.deepcopy(instance)
        
        attack_args = prepare_attack_call_args(instance, args.attack)
        adversarial_prompt = attack.transform(**attack_args)[0]
        
        new_instance.surrogate_model = args.surrogate_model
        new_instance.attack = args.attack
        new_instance.adversarial_prompt = adversarial_prompt
        new_instances.append(new_instance)
        
    write_jsonl([instance.to_dict() for instance in new_instances], args.output_file)