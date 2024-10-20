# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

ADE20K_COLORMAP = [
    (0, 0, 0),
    (120, 120, 120),
    (180, 120, 120),
    (6, 230, 230),
    (80, 50, 50),
    (4, 200, 3),
    (120, 120, 80),
    (140, 140, 140),
    (204, 5, 255),
    (230, 230, 230),
    (4, 250, 7),
    (224, 5, 255),
    (235, 255, 7),
    (150, 5, 61),
    (120, 120, 70),
    (8, 255, 51),
    (255, 6, 82),
    (143, 255, 140),
    (204, 255, 4),
    (255, 51, 7),
    (204, 70, 3),
    (0, 102, 200),
    (61, 230, 250),
    (255, 6, 51),
    (11, 102, 255),
    (255, 7, 71),
    (255, 9, 224),
    (9, 7, 230),
    (220, 220, 220),
    (255, 9, 92),
    (112, 9, 255),
    (8, 255, 214),
    (7, 255, 224),
    (255, 184, 6),
    (10, 255, 71),
    (255, 41, 10),
    (7, 255, 255),
    (224, 255, 8),
    (102, 8, 255),
    (255, 61, 6),
    (255, 194, 7),
    (255, 122, 8),
    (0, 255, 20),
    (255, 8, 41),
    (255, 5, 153),
    (6, 51, 255),
    (235, 12, 255),
    (160, 150, 20),
    (0, 163, 255),
    (140, 140, 140),
    (250, 10, 15),
    (20, 255, 0),
    (31, 255, 0),
    (255, 31, 0),
    (255, 224, 0),
    (153, 255, 0),
    (0, 0, 255),
    (255, 71, 0),
    (0, 235, 255),
    (0, 173, 255),
    (31, 0, 255),
    (11, 200, 200),
    (255, 82, 0),
    (0, 255, 245),
    (0, 61, 255),
    (0, 255, 112),
    (0, 255, 133),
    (255, 0, 0),
    (255, 163, 0),
    (255, 102, 0),
    (194, 255, 0),
    (0, 143, 255),
    (51, 255, 0),
    (0, 82, 255),
    (0, 255, 41),
    (0, 255, 173),
    (10, 0, 255),
    (173, 255, 0),
    (0, 255, 153),
    (255, 92, 0),
    (255, 0, 255),
    (255, 0, 245),
    (255, 0, 102),
    (255, 173, 0),
    (255, 0, 20),
    (255, 184, 184),
    (0, 31, 255),
    (0, 255, 61),
    (0, 71, 255),
    (255, 0, 204),
    (0, 255, 194),
    (0, 255, 82),
    (0, 10, 255),
    (0, 112, 255),
    (51, 0, 255),
    (0, 194, 255),
    (0, 122, 255),
    (0, 255, 163),
    (255, 153, 0),
    (0, 255, 10),
    (255, 112, 0),
    (143, 255, 0),
    (82, 0, 255),
    (163, 255, 0),
    (255, 235, 0),
    (8, 184, 170),
    (133, 0, 255),
    (0, 255, 92),
    (184, 0, 255),
    (255, 0, 31),
    (0, 184, 255),
    (0, 214, 255),
    (255, 0, 112),
    (92, 255, 0),
    (0, 224, 255),
    (112, 224, 255),
    (70, 184, 160),
    (163, 0, 255),
    (153, 0, 255),
    (71, 255, 0),
    (255, 0, 163),
    (255, 204, 0),
    (255, 0, 143),
    (0, 255, 235),
    (133, 255, 0),
    (255, 0, 235),
    (245, 0, 255),
    (255, 0, 122),
    (255, 245, 0),
    (10, 190, 212),
    (214, 255, 0),
    (0, 204, 255),
    (20, 0, 255),
    (255, 255, 0),
    (0, 153, 255),
    (0, 41, 255),
    (0, 255, 204),
    (41, 0, 255),
    (41, 255, 0),
    (173, 0, 255),
    (0, 245, 255),
    (71, 0, 255),
    (122, 0, 255),
    (0, 255, 184),
    (0, 92, 255),
    (184, 255, 0),
    (0, 133, 255),
    (255, 214, 0),
    (25, 194, 194),
    (102, 255, 0),
    (92, 0, 255),
]

ADE20K_CLASS_NAMES = [
    "",
    "wall",
    "building;edifice",
    "sky",
    "floor;flooring",
    "tree",
    "ceiling",
    "road;route",
    "bed",
    "windowpane;window",
    "grass",
    "cabinet",
    "sidewalk;pavement",
    "person;individual;someone;somebody;mortal;soul",
    "earth;ground",
    "door;double;door",
    "table",
    "mountain;mount",
    "plant;flora;plant;life",
    "curtain;drape;drapery;mantle;pall",
    "chair",
    "car;auto;automobile;machine;motorcar",
    "water",
    "painting;picture",
    "sofa;couch;lounge",
    "shelf",
    "house",
    "sea",
    "mirror",
    "rug;carpet;carpeting",
    "field",
    "armchair",
    "seat",
    "fence;fencing",
    "desk",
    "rock;stone",
    "wardrobe;closet;press",
    "lamp",
    "bathtub;bathing;tub;bath;tub",
    "railing;rail",
    "cushion",
    "base;pedestal;stand",
    "box",
    "column;pillar",
    "signboard;sign",
    "chest;of;drawers;chest;bureau;dresser",
    "counter",
    "sand",
    "sink",
    "skyscraper",
    "fireplace;hearth;open;fireplace",
    "refrigerator;icebox",
    "grandstand;covered;stand",
    "path",
    "stairs;steps",
    "runway",
    "case;display;case;showcase;vitrine",
    "pool;table;billiard;table;snooker;table",
    "pillow",
    "screen;door;screen",
    "stairway;staircase",
    "river",
    "bridge;span",
    "bookcase",
    "blind;screen",
    "coffee;table;cocktail;table",
    "toilet;can;commode;crapper;pot;potty;stool;throne",
    "flower",
    "book",
    "hill",
    "bench",
    "countertop",
    "stove;kitchen;stove;range;kitchen;range;cooking;stove",
    "palm;palm;tree",
    "kitchen;island",
    "computer;computing;machine;computing;device;data;processor;electronic;computer;information;processing;system",
    "swivel;chair",
    "boat",
    "bar",
    "arcade;machine",
    "hovel;hut;hutch;shack;shanty",
    "bus;autobus;coach;charabanc;double-decker;jitney;motorbus;motorcoach;omnibus;passenger;vehicle",
    "towel",
    "light;light;source",
    "truck;motortruck",
    "tower",
    "chandelier;pendant;pendent",
    "awning;sunshade;sunblind",
    "streetlight;street;lamp",
    "booth;cubicle;stall;kiosk",
    "television;television;receiver;television;set;tv;tv;set;idiot;box;boob;tube;telly;goggle;box",
    "airplane;aeroplane;plane",
    "dirt;track",
    "apparel;wearing;apparel;dress;clothes",
    "pole",
    "land;ground;soil",
    "bannister;banister;balustrade;balusters;handrail",
    "escalator;moving;staircase;moving;stairway",
    "ottoman;pouf;pouffe;puff;hassock",
    "bottle",
    "buffet;counter;sideboard",
    "poster;posting;placard;notice;bill;card",
    "stage",
    "van",
    "ship",
    "fountain",
    "conveyer;belt;conveyor;belt;conveyer;conveyor;transporter",
    "canopy",
    "washer;automatic;washer;washing;machine",
    "plaything;toy",
    "swimming;pool;swimming;bath;natatorium",
    "stool",
    "barrel;cask",
    "basket;handbasket",
    "waterfall;falls",
    "tent;collapsible;shelter",
    "bag",
    "minibike;motorbike",
    "cradle",
    "oven",
    "ball",
    "food;solid;food",
    "step;stair",
    "tank;storage;tank",
    "trade;name;brand;name;brand;marque",
    "microwave;microwave;oven",
    "pot;flowerpot",
    "animal;animate;being;beast;brute;creature;fauna",
    "bicycle;bike;wheel;cycle",
    "lake",
    "dishwasher;dish;washer;dishwashing;machine",
    "screen;silver;screen;projection;screen",
    "blanket;cover",
    "sculpture",
    "hood;exhaust;hood",
    "sconce",
    "vase",
    "traffic;light;traffic;signal;stoplight",
    "tray",
    "ashcan;trash;can;garbage;can;wastebin;ash;bin;ash-bin;ashbin;dustbin;trash;barrel;trash;bin",
    "fan",
    "pier;wharf;wharfage;dock",
    "crt;screen",
    "plate",
    "monitor;monitoring;device",
    "bulletin;board;notice;board",
    "shower",
    "radiator",
    "glass;drinking;glass",
    "clock",
    "flag",
]


VOC2012_COLORMAP = [
    (0, 0, 0),
    (128, 0, 0),
    (0, 128, 0),
    (128, 128, 0),
    (0, 0, 128),
    (128, 0, 128),
    (0, 128, 128),
    (128, 128, 128),
    (64, 0, 0),
    (192, 0, 0),
    (64, 128, 0),
    (192, 128, 0),
    (64, 0, 128),
    (192, 0, 128),
    (64, 128, 128),
    (192, 128, 128),
    (0, 64, 0),
    (128, 64, 0),
    (0, 192, 0),
    (128, 192, 0),
    (0, 64, 128),
]


VOC2012_CLASS_NAMES = [
    "",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]
