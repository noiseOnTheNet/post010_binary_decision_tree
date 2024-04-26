use std::fmt::Debug;

#[derive(Debug)]
pub struct Node<T> {
    value: T,
    left: Option<Box<Node<T>>>,
    right: Option<Box<Node<T>>>,
}

impl<T> Node<T> {
    pub fn new(value: T) -> Node<T> {
        Node {
            value,
            left: None,
            right: None,
        }
    }
}

impl<T> From<Node<T>> for Option<Box<Node<T>>> {
    fn from(value: Node<T>) -> Self {
        Some(Box::new(value))
    }
}

#[derive(Debug, Default)]
pub struct Tree<T> {
    root: Option<Box<Node<T>>>,
}

impl<T> Tree<T> {
    pub fn new() -> Tree<T> {
        Tree { root: None }
    }

    pub fn from_node(node: Node<T>) -> Tree<T> {
        Tree {
            root: Some(node.into()),
        }
    }
}

impl<T: Ord> Tree<T> {
    fn insert(&mut self, value: T) {
        match self.root {
            None => {
                self.root = Node::new(value).into();
            }
            Some(ref mut node) => {
                Tree::<T>::insert_recursive(node, value);
            }
        }
    }

    fn insert_recursive(node: &mut Node<T>, value: T) {
        if value > node.value {
            match node.right {
                None => {
                    node.right = Node::new(value).into();
                }
                Some(ref mut n) => {
                    Tree::<T>::insert_recursive(n, value);
                }
            }
        } else if value < node.value {
            match node.left {
                None => {
                    node.left = Node::new(value).into();
                }
                Some(ref mut n) => {
                    Tree::<T>::insert_recursive(n, value);
                }
            }
        }
    }
}

#[derive(Debug, Copy, Clone)]
enum Address {
    Enter,
    LeftCompleted,
    ValueYield,
    Completed,
}

pub struct TreeIter<'a, T> {
    stack: Vec<(Address, &'a Node<T>)>,
}

impl<'a, T> TreeIter<'a, T> {
    fn new(tree: &'a Tree<T>) -> TreeIter<'a, T> {
        match tree.root {
            None => TreeIter { stack: Vec::new() },
            Some(ref node) => TreeIter {
                stack: vec![(Address::Enter, &node)],
            },
        }
    }
    fn next_item(&mut self) -> Option<&'a T> {
        while let Some((address, node)) = self.stack.pop() {
            match address {
                Address::Enter => match node.left {
                    None => {
                        self.stack.push((Address::LeftCompleted, node));
                    }
                    Some(ref left) => {
                        self.stack.push((Address::LeftCompleted, node));
                        self.stack.push((Address::Enter, left));
                    }
                },
                Address::LeftCompleted => {
                    self.stack.push((Address::ValueYield, node));
                    return Some(&node.value);
                }
                Address::ValueYield => match node.right {
                    None => {
                        self.stack.push((Address::Completed, node));
                    }
                    Some(ref right) => {
                        self.stack.push((Address::Completed, node));
                        self.stack.push((Address::Enter, right));
                    }
                },
                Address::Completed => {}
            }
        }
        None
    }
}

impl<'a, T> Iterator for TreeIter<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        self.next_item()
    }
}

impl<'a, T> IntoIterator for &'a Tree<T> {
    type Item = &'a T;
    type IntoIter = TreeIter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        TreeIter::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_a_root_node() {
        let mut tree: Tree<i64> = Tree::new();
        tree.insert(8);
        tree.insert(10);
        tree.insert(4);
        tree.insert(6);
        tree.insert(5);
        println!("{:?}", tree);
        let result: Vec<i64> = tree.into_iter().map(|x| (*x).clone()).collect();
        assert_eq!(result, vec![4, 5, 6, 8, 10]);
    }
}
