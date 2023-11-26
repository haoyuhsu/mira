# coding=utf-8
# Copyright 2021 The Ravens Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ravens tasks."""

from ravens.tasks.align_box_corner import AlignBoxCorner
from ravens.tasks.assembling_kits import AssemblingKits
from ravens.tasks.assembling_kits import AssemblingKitsEasy
from ravens.tasks.block_insertion import BlockInsertion, BlockInsertionSixDofDiscrete
from ravens.tasks.block_insertion import BlockInsertionEasy
from ravens.tasks.block_insertion import BlockInsertionNoFixture
from ravens.tasks.block_insertion import BlockInsertionSixDof, BlockInsertionSixDofOOD
from ravens.tasks.block_insertion import BlockInsertionFiveDofDiscrete, BlockInsertionSixDofPerspective
from ravens.tasks.block_insertion import BlockInsertionTranslation
from ravens.tasks.manipulating_rope import ManipulatingRope
from ravens.tasks.packing_shoes import PackingShoes
from ravens.tasks.packing_shapes import PackingShapes
from ravens.tasks.packing_clothes import PackingClothes
from ravens.tasks.packing_ropes import PackingRopes
from ravens.tasks.place_red_in_green import PlaceRedInGreen, PlaceRedInGreenSixDofDiscrete, PlaceRedInGreenSixDof, PlaceRedInGreenSixDofOOD
from ravens.tasks.sweeping_piles import SweepingPiles
from ravens.tasks.task import Task
from ravens.tasks.towers_of_hanoi import TowersOfHanoi
from ravens.tasks.hanging_disks import HangingDisks, HangingDisksOOD
from ravens.tasks.hanging_ring import HangingRing
from ravens.tasks.stacking_kits import StackingKits, StackingKitsOOD
from ravens.tasks.palletizing_boxes import PalletizingBoxes

from ravens.tasks.stack_block_pyramid import StackBlockPyramid, StackingTowers
from ravens.tasks.packing_boxes import PackingBoxes, StackingBoxes

names = {
    'align-box-corner': AlignBoxCorner,
    'assembling-kits': AssemblingKits,   # put letter-shaped blocks into a board with holes
    'assembling-kits-easy': AssemblingKitsEasy,
    'block-insertion': BlockInsertion,   # put 'L' shaped block into a fixture
    'block-insertion-easy': BlockInsertionEasy,
    'block-insertion-nofixture': BlockInsertionNoFixture,
    'block-insertion-sixdof': BlockInsertionSixDof,
    'block-insertion-sixdof-perspective': BlockInsertionSixDofPerspective,
    'block-insertion-sixdof-ood': BlockInsertionSixDofOOD,
    'block-insertion-sixdof-discrete': BlockInsertionSixDofDiscrete,
    'block-insertion-fivedof-discrete': BlockInsertionFiveDofDiscrete,
    'block-insertion-translation': BlockInsertionTranslation,
    'manipulating-rope': ManipulatingRope,
    'packing-boxes': PackingBoxes,        # put boxes into a container (useful for our case)
    'packing-shoes': PackingShoes,
    'packing-shapes': PackingShapes,
    'palletizing-boxes': PalletizingBoxes,    # put a box on a wooden pallet one by one (useful for our case)
    'place-red-in-green': PlaceRedInGreen,    # put red block into green bowl
    'place-red-in-green-sixdof-discrete': PlaceRedInGreenSixDofDiscrete,
    'place-red-in-green-sixdof': PlaceRedInGreenSixDof,
    'place-red-in-green-sixdof-ood': PlaceRedInGreenSixDofOOD,
    'stack-block-pyramid': StackBlockPyramid,   # stack blocks into a pyramid (1, 2, 3)
    'sweeping-piles': SweepingPiles,
    'towers-of-hanoi': TowersOfHanoi,
    'hanging-disks': HangingDisks,
    'hanging-disks-ood': HangingDisksOOD,
    'packing-clothes': PackingClothes,
    'packing-ropes': PackingRopes,
    'hanging-ring': HangingRing,
    'stacking-kits': StackingKits,         # stacking a kit on top of another kit
    'stacking-kits-ood': StackingKitsOOD,
    'stack-boxes': StackingBoxes,          # our project task
    'stack-towers': StackingTowers,        # our project task
}
