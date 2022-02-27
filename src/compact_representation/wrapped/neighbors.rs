//! This module holds the `NeighborDeterminableGame` trait impl for wrapped and other code related
//! to its SIMD implementation.
use std::convert::TryInto;

use core_simd::{simd_swizzle, Simd};

use crate::types::{Move, NeighborDeterminableGame, PositionGettableGame};

use super::super::core::CellIndex;
use super::super::CellNum as CN;
use super::CellBoard;

/// a game for which the neighbors of a given Position can be determined
pub trait FixedNeighborDeterminableGame<const N_MOVES: usize>: PositionGettableGame {
    /// returns the neighboring positions
    fn neighbors_fixed<'a>(
        &'a self,
        pos: &Self::NativePositionType,
    ) -> [Self::NativePositionType; N_MOVES];

    /// returns the neighboring positions, and the Move required to get to each
    fn possible_moves_fixed<'a>(
        &'a self,
        pos: &Self::NativePositionType,
    ) -> [(Move, Self::NativePositionType); N_MOVES];
}

impl<T: CN, const BOARD_SIZE: usize, const MAX_SNAKES: usize> FixedNeighborDeterminableGame<4>
    for CellBoard<T, BOARD_SIZE, MAX_SNAKES>
{
    fn possible_moves_fixed<'a>(
        &'a self,
        pos: &Self::NativePositionType,
    ) -> [(Move, Self::NativePositionType); 4] {
        let width = self.embedded.get_actual_width();
        let width_i: i8 = width.try_into().unwrap();
        let head_pos = pos.into_position(width);

        let move_simd = Move::all_simd();
        let current_pos_simd = Simd::<i8, 2>::from_array([head_pos.x as i8, head_pos.y as i8]);
        let current_pos_simd = simd_swizzle!(current_pos_simd, [0, 1, 0, 1, 0, 1, 0, 1]);

        let new_pos_simd = current_pos_simd + move_simd;

        let negative_overflow_simd = Simd::<i8, 8>::splat(-1);
        let negative_overflow_mask = new_pos_simd.lanes_eq(negative_overflow_simd);

        let positive_overflow_simd = Simd::<i8, 8>::splat(width_i);
        let positive_overflow_mask = new_pos_simd.lanes_eq(positive_overflow_simd);

        let new_pos_simd = negative_overflow_mask.select(Simd::splat(width_i - 1), new_pos_simd);
        let new_pos_simd = positive_overflow_mask.select(Simd::splat(0), new_pos_simd);
        let new_pos_simd: Simd<u8, 8> = new_pos_simd.cast();

        let x_values = simd_swizzle!(new_pos_simd, [0, 2, 4, 6]);
        let mut y_values = simd_swizzle!(new_pos_simd, [1, 3, 5, 7]);
        y_values *= Simd::splat(width as u8);

        let indices = x_values + y_values;
        let indices = indices
            .to_array()
            .map(|idx| CellIndex::from_u32(idx.into()));
        let all = Move::all();

        all.zip(indices)
    }

    fn neighbors_fixed<'a>(
        &'a self,
        pos: &Self::NativePositionType,
    ) -> [Self::NativePositionType; 4] {
        self.possible_moves_fixed(pos).map(|(_, ci)| ci)
    }
}

impl<T> NeighborDeterminableGame for T
where
    T: FixedNeighborDeterminableGame<4>,
{
    fn neighbors<'a>(
        &'a self,
        pos: &Self::NativePositionType,
    ) -> Box<dyn Iterator<Item = Self::NativePositionType> + 'a> {
        Box::new(IntoIterator::into_iter(self.neighbors_fixed(pos)))
    }

    fn possible_moves<'a>(
        &'a self,
        pos: &Self::NativePositionType,
    ) -> Box<dyn Iterator<Item = (Move, Self::NativePositionType)> + 'a> {
        Box::new(IntoIterator::into_iter(self.possible_moves_fixed(pos)))
    }
}

#[cfg(test)]
mod test {
    use crate::{
        game_fixture,
        types::{build_snake_id_map, HeadGettableGame, Move, NeighborDeterminableGame, SnakeId},
    };

    use super::super::{CellBoard4Snakes11x11, CellIndex};

    #[test]
    fn test_neighbors_and_possible_moves_cornered() {
        let g = game_fixture(include_str!("../../../fixtures/cornered_wrapped.json"));
        let snake_id_mapping = build_snake_id_map(&g);
        let compact: CellBoard4Snakes11x11 = g.as_wrapped_cell_board(&snake_id_mapping).unwrap();

        let head = compact.get_head_as_native_position(&SnakeId(0));
        assert_eq!(head, CellIndex(10 * 11));

        let expected_possible_moves = vec![
            (Move::Up, CellIndex(0)),
            (Move::Down, CellIndex(9 * 11)),
            (Move::Left, CellIndex(10 * 11 + 10)),
            (Move::Right, CellIndex(10 * 11 + 1)),
        ];

        assert_eq!(
            compact.possible_moves(&head).collect::<Vec<_>>(),
            expected_possible_moves
        );

        assert_eq!(
            compact.neighbors(&head).collect::<Vec<_>>(),
            expected_possible_moves
                .into_iter()
                .map(|(_, pos)| pos)
                .collect::<Vec<_>>()
        );
    }
}
