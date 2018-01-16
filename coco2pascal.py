import baker
import json
#from path import path
from cytoolz import merge, join, groupby
from cytoolz.compatibility import iteritems
from cytoolz.curried import update_in
from itertools import starmap
from collections import deque
from lxml import etree, objectify
from scipy.io import savemat
from scipy.ndimage import imread
import os
import pandas as pd

def keyjoin(leftkey, leftseq, rightkey, rightseq):
    return starmap(merge, join(leftkey, leftseq, rightkey, rightseq))


def root(folder, filename, width, height):
    E = objectify.ElementMaker(annotate=False)
    return E.annotation(
            E.folder(folder),
            E.filename(filename),
            E.source(
                E.database('MS COCO 2014'),
                E.annotation('MS COCO 2014'),
                E.image('Flickr'),
                ),
            E.size(
                E.width(width),
                E.height(height),
                E.depth(3),
                ),
            E.segmented(0)
            )


def instance_to_xml(anno):
    E = objectify.ElementMaker(annotate=False)
    xmin, ymin, width, height = anno['bbox']
    return E.object(
            E.name(anno['category_id']),
            E.bndbox(
                E.xmin(xmin),
                E.ymin(ymin),
                E.xmax(xmin+width),
                E.ymax(ymin+height),
                ),
            )


@baker.command
def write_categories(coco_annotation, dst):
    '''

    :param coco_annotation: annotation file of coco
    :param dst:
    :return:
    '''
    content = json.loads(coco_annotation)
    categories = tuple( d['name'] for d in content['categories'])
    savemat(dst, {'categories': categories})


def get_instances(coco_annotation):
    print("coco annotation file:\t",coco_annotation)
    with open(coco_annotation) as json_file:
        content = json.load(json_file)
    #content = json.loads(coco_annotation)
        categories = {d['id']: d['name'] for d in content['categories']}
        return categories, tuple(keyjoin('id', content['images'], 'image_id', content['annotations']))


@baker.command
def create_imageset(annotations, dst):
    val_txt = dst / 'val.txt'
    train_txt = dst / 'train.txt'

    for val in annotations.listdir('*val*'):
        val_txt.write_text('{}\n'.format(val.basename().stripext()), append=True)

    for train in annotations.listdir('*train*'):
        train_txt.write_text('{}\n'.format(train.basename().stripext()), append=True)

@baker.command
def create_annotations(dbpath, subset, dst):
    '''

    :param dbpath:   root path of coco dataset
    :param subset:  'train' or  'val'
    :param dst:     where to save transfered result
    :return:
    '''
    annotations_path =  dbpath+ '/annotations_trainval2014/annotations/instances_{}2014.json'.format(subset)
    images_path = dbpath + '/images/{}2014'.format(subset)
    categories , instances= get_instances(annotations_path)

    if not os.path.exists(dst):
        os.makedirs(dst)

    for i, instance in enumerate(instances):
        instances[i]['category_id'] = categories[instance['category_id']]

    for name, group in iteritems(groupby('file_name', instances)):
        print("image_path is %s , name is %s "%(images_path,name))
        img = imread(images_path+"/"+name)
        if img.ndim == 3:
            annotation = root(images_path, name,
                              group[0]['height'], group[0]['width'])
            for instance in group:
                annotation.append(instance_to_xml(instance))
            etree.ElementTree(annotation).write(dst+  '/{}.xml'.format(name.split(".")[0]))
            print(name)
        else:
            print(instance['file_name'])



if __name__ == '__main__':
    #baker.run()
    dbpath = "d:/data/coco"
    subset="val"
    dst = "d:/data/coco/pascal_format/val"
    create_annotations(dbpath,subset,dst)
    # json_file = "d:/data/coco/annotations_trainval2014/annotations/instances_train2014.json"
    # df =  pd.read_json(json_file)
    # print(df[:10])