#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import sys
from io import open

import numpy as np
import torch
import json

from torch.utils.data import (DataLoader, SequentialSampler, RandomSampler, TensorDataset)
from tqdm import tqdm, trange

import ray
from ray import tune
from ray.tune.schedulers import HyperBandScheduler

from models.modeling_bert import QuestionAnswering, Config
from utils.optimization import AdamW, WarmupLinearSchedule
from utils.tokenization import BertTokenizer
from utils.korquad_utils import (read_squad_examples, convert_examples_to_features, RawResult, write_predictions)
from debug.evaluate_korquad import evaluate as korquad_eval

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


# In[2]:


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


# In[3]:


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# In[4]:


from ray import tune
from ray.tune import track
from ray.tune.schedulers import HyperBandScheduler
from ray.tune.suggest.bayesopt import BayesOptSearch


ray.shutdown()
ray.init(webui_host='127.0.0.1')


# In[5]:


search_space = {
    "max_seq_length": 512,
    "doc_stride": 128,
    "max_query_length": tune.sample_from(lambda _: int(np.random.uniform(50, 100))), #tune.uniform(50, 100),
    "train_batch_size": 32,
    "learning_rate": tune.loguniform(5e-4, 5e-7, 10),
    "num_train_epochs": tune.grid_search([4, 8, 12, 16]),
    "max_grad_norm": 1.0,
    "adam_epsilon": 1e-6,
    "warmup_proportion": 0.1,
    "n_best_size": tune.sample_from(lambda _: int(np.random.uniform(50, 100))), #tune.uniform(50, 100),
    "max_answer_length": tune.sample_from(lambda _: int(np.random.uniform(12, 25))), #tune.uniform(12, 25), 
    "seed": tune.sample_from(lambda _: int(np.random.uniform(1e+6, 1e+8)))
}


# In[ ]:


def load_and_cache_examples(predict_file, max_seq_length, doc_stride, max_query_length, tokenizer):
    # Load data features from cache or dataset file
    examples = read_squad_examples(input_file=predict_file,
                                   is_training=False,
                                   version_2_with_negative=False)
    
    features = convert_examples_to_features(examples=examples,
                                            tokenizer=tokenizer,
                                            max_seq_length=max_seq_length,
                                            doc_stride=doc_stride,
                                            max_query_length=max_query_length,
                                            is_training=False)
    return examples, features


# In[ ]:


def evaluate(predict_file, batch_size, device, output_dir, n_best_size, max_answer_length, model, eval_examples, eval_features):
    """ Eval """
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    logger.info("***** Evaluating *****")
    logger.info("  Num features = %d", len(dataset))
    logger.info("  Batch size = %d", batch_size)

    model.eval()
    all_results = []
#     set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    logger.info("Start evaluating!")
    for input_ids, input_mask, segment_ids, example_indices in tqdm(dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        with torch.no_grad():
            batch_start_logits, batch_end_logits = model(input_ids, segment_ids, input_mask)
        for i, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            all_results.append(RawResult(unique_id=unique_id,
                                         start_logits=start_logits,
                                         end_logits=end_logits))
    output_prediction_file = os.path.join(output_dir, "predictions.json")
    output_nbest_file = os.path.join(output_dir, "nbest_predictions.json")
    write_predictions(eval_examples, eval_features, all_results,
                      n_best_size, max_answer_length,
                      False, output_prediction_file, output_nbest_file,
                      None, False, False, 0.0)

    expected_version = 'KorQuAD_v1.0'
    with open(predict_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        read_version = "_".join(dataset_json['version'].split("_")[:-1])
        if (read_version != expected_version):
            logger.info('Evaluation expects ' + expected_version + ', but got dataset with ' + read_version, file=sys.stderr)
        dataset = dataset_json['data']
    with open(os.path.join(output_dir, "predictions.json")) as prediction_file:
        predictions = json.load(prediction_file)
    _eval = korquad_eval(dataset, predictions)
    logger.info(json.dumps(_eval))

    return _eval


# In[6]:


def train_korquad(train_config):
    
    # setup
    basepath = '/jupyterhome/enpline_bert_competition/korquad-challenge/src'
        
    logger.info("train_config : %s" % str(train_config))

    output_dir='output'
    checkpoint=os.path.join(basepath,'data/bert_small_ckpt.bin')
    model_config=os.path.join(basepath,'data/bert_small.json')
    vocab_file=os.path.join(basepath,'data/ko_vocab_32k.txt')
    train_file=os.path.join(basepath, 'data/KorQuAD_v1.0_train.json')
    predict_file=os.path.join(basepath, 'data/KorQuAD_v1.0_dev.json')
        
    
    null_score_diff_threshold = 0.0
    no_cuda = False
    verbose_logging = False
    fp16 = True
    fp16_opt_level = 'O2' 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("device: {} n_gpu: {}, 16-bits training: {}".format(device, n_gpu, fp16))

    random.seed(train_config['seed'])
    np.random.seed(train_config['seed'])
    torch.manual_seed(train_config['seed'])
    if n_gpu > 0:
        torch.cuda.manual_seed_all(train_config['seed'])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tokenizer = BertTokenizer(vocab_file, max_len=train_config['max_seq_length'], do_basic_tokenize=True)
    
    # Prepare model
    config = Config.from_json_file(model_config)
    model = QuestionAnswering(config)
    model.bert.load_state_dict(torch.load(checkpoint))
    num_params = count_parameters(model)
    logger.info("Total Parameter: %d" % num_params)
    logger.info("Hyper-parameters: %s" % str(train_config))
    paramfile_path = os.path.join(output_dir, 'hyperparameters.txt')
    
    with open(paramfile_path, "w") as paramfile:
        logger.info("writing hyperparameters at",paramfile_path)
        paramfile.write("%s" % str(train_config))

    model.to(device)

    cached_train_features_file = train_file + '_{0}_{1}_{2}'.format(str(train_config['max_seq_length']), str(train_config['doc_stride']),
                                                                         str(train_config['max_query_length']))
    train_examples = read_squad_examples(input_file=train_file, is_training=True, version_2_with_negative=False)
    
    try:
        with open(cached_train_features_file, "rb") as reader:
            train_features = pickle.load(reader)
    except:
        train_features = convert_examples_to_features(
            examples=train_examples,
            tokenizer=tokenizer,
            max_seq_length=train_config['max_seq_length'],
            doc_stride=train_config['doc_stride'],
            max_query_length=train_config['max_query_length'],
            is_training=True)
        logger.info("  Saving train features into cached file %s", cached_train_features_file)
        with open(cached_train_features_file, "wb") as writer:
            pickle.dump(train_features, writer)

    num_train_optimization_steps = int(len(train_features) / train_config['train_batch_size']) * train_config['num_train_epochs']

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=train_config['learning_rate'],
                      eps=train_config['adam_epsilon'])
    scheduler = WarmupLinearSchedule(optimizer,
                                     warmup_steps=num_train_optimization_steps*0.1,
                                     t_total=num_train_optimization_steps)

    if fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)

    logger.info("***** Running training *****")
    logger.info("  Num orig examples = %d", len(train_examples))
    logger.info("  Num split examples = %d", len(train_features))
    logger.info("  Batch size = %d", train_config['train_batch_size'])
    logger.info("  Num steps = %d", num_train_optimization_steps)
    num_train_step = num_train_optimization_steps

    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
    all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                               all_start_positions, all_end_positions)

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_config['train_batch_size'])

    model.train()
    global_step = 0
    epoch = 0
    
    output_model_file = ''
    
    # training
#     for epoch_idx in trange(int(train_config['num_train_epochs'])):
#         iter_bar = tqdm(train_dataloader, desc="Train(XX Epoch) Step(XX/XX) (Mean loss=X.X) (loss=X.X)")

    for epoch_idx in range(int(train_config['num_train_epochs'])):
        tr_step, total_loss, mean_loss = 0, 0., 0.
        for step, batch in enumerate(train_dataloader):
            if n_gpu == 1:
                batch = tuple(t.to(device) for t in batch)  # multi-gpu does scattering it-self
            input_ids, input_mask, segment_ids, start_positions, end_positions = batch
            loss = model(input_ids, segment_ids, input_mask, start_positions, end_positions)
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), train_config['max_grad_norm'])
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config['max_grad_norm'])

            scheduler.step()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            tr_step += 1
            total_loss += loss
            mean_loss = total_loss / tr_step
#             iter_bar.set_description("Train Step(%d / %d) (Mean loss=%5.5f) (loss=%5.5f)" %
#                                      (global_step, num_train_step, mean_loss, loss.item()))

            epoch += 1 
            
    logger.info("** ** * Saving file * ** **")  
    model_checkpoint = "korquad_%d.bin" % (epoch)
    logger.info(model_checkpoint)
        
    #save the last model
    output_model_file = os.path.join(output_dir, model_checkpoint)
    if n_gpu > 1:
        torch.save(model.module.state_dict(), output_model_file)
    else:
        torch.save(model.state_dict(), output_model_file)
      
    # Evaluate with final model
    examples, features = load_and_cache_examples(predict_file, train_config['max_seq_length'], train_config['doc_stride'], 
                                                 train_config['max_query_length'], tokenizer)
    eval = evaluate(predict_file=predict_file, batch_size=16, device=device, output_dir=output_dir, n_best_size=train_config['n_best_size'], max_answer_length=train_config['max_answer_length'],
             model=model, eval_examples=examples, eval_features=features)
    
    logger.info("-" * 16, 'evaltion', "-" *  16)
    logger.info(eval)

    track.log(f1 = eval['f1'])
    


# In[ ]:


analysis = tune.run(train_korquad, config=search_space, scheduler=HyperBandScheduler(metric='f1', mode='max'), resources_per_trial={'gpu':1})


# In[ ]:


dfs = analysis.trial_dataframes


# In[ ]:


# ax = None
# for d in dfs.values():
#     ax = d.mean_loss.plot(ax=ax, legend=True)    
    
# ax.set_xlabel("Epochs")
# ax.set_ylabel("Mean Loss")

