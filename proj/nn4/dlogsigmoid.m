function [ dlogsig ] = dlogsigmoid(logsig)
%DLOGSIGMOID Summary of this function goes here
%   Detailed explanation goes here

dlogsig = logsig.*(1-logsig);