'''
Created by Minjuc on Mar.13.2023
'''
import torch


def main():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
    print('Hello world')


if __name__ == '__main__':
    main()
