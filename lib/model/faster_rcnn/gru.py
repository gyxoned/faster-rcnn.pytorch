#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import cPickle
import numpy as np

# hidden_states = feature
class GRUMessage(nn.Module):
	def __init__(self, feature_size, score_size, input_size, class_agnostic=False, iters=2):
		super(GRUMessage, self).__init__()
		self.iters = iters
		self.input_size = input_size
		if class_agnostic:
			bbox_num = 4 * score_size
		else:
			bbox_num = 4

		self.fc_score = nn.Linear(score_size,feature_size)
		self.fc_box = nn.Linear(bbox_num,feature_size)
		self.fc_input = nn.Linear(feature_size,input_size)

		self.gru_cell = nn.GRUCell(input_size,feature_size)
		self.classifer = nn.Linear(feature_size,score_size)
		self.bbox_prec = nn.Linear(feature_size,bbox_num)

	def forward(self, features, scores, boxes):
		new_scores = []
		new_boxes = []
		for im in range(features.size(0)):
			rois_sum = features[im].size(0)
			h_output = []
			s_output = []
			b_output = []
			h_output.append(features[im])
			s_output.append(scores[im])
			b_output.append(boxes[im])

			for iter in xrange(self.iters):
				h = self.fc_input(F.relu(h_output[iter] * self.fc_score(s_output[iter]) * self.fc_box(b_output[iter])))
				input_sum = torch.sum(h,0).view(1,self.input_size).expand(rois_sum,self.input_size)
				input = (input_sum - h) / max(1,(rois_sum-1))
				hx = self.gru_cell(input,h_output[iter])
				h_output.append(hx)
				scores_gru = F.softmax(self.classifer(h_output[-1]))
				bboxes_gru = self.bbox_prec(h_output[-1])
				s_output.append(scores_gru)
				b_output.append(bboxes_gru)

			new_scores.append(s_output[-1])
			new_boxes.append(b_output[-1])

		new_scores = torch.cat(new_scores)
		new_boxes = torch.cat(new_boxes)

		return new_scores, new_boxes