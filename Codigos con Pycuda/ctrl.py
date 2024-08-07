from __future__ import print_function, division

import cupy as cp
import networkx as nx
from warnings import warn

from .fuzzymath.fuzzy_ops import _interp_universe_fast
from skfuzzy import interp_membership, defuzz
from .fuzzyvariable import FuzzyVariable
from antecedent_consecuent import Antecedent, Consequent
from term import Term, WeightedTerm, TermAggregate
from .rule import Rule

try:
    from collections import OrderedDict
except ImportError:
    from ordered_dict import OrderedDict

class ControlSystem(object):
    def __init__(self, rules=None):
        self.graph = nx.DiGraph()
        self._rule_generator = RuleOrderGenerator(self)
        if rules is not None:
            if hasattr(rules, '__iter__'):
                for rule in rules:
                    self.addrule(rule)
            else:
                try:
                    self.addrule(rules)
                except:
                    raise ValueError("Optional argument `rules` must be a FuzzyRule or iterable of FuzzyRules.")

    @property
    def rules(self):
        return self._rule_generator

    @property
    def antecedents(self):
        for node in self.graph.nodes():
            if isinstance(node, Antecedent):
                yield node

    @property
    def consequents(self):
        for node in self.graph.nodes():
            if isinstance(node, Consequent):
                yield node

    @property
    def fuzzy_variables(self):
        for node in self.graph.nodes():
            if isinstance(node, FuzzyVariable):
                yield node

    def addrule(self, rule):
        if not isinstance(rule, Rule):
            raise ValueError("Input rule must be a Rule object!")
        labels = []
        for r in self.rules:
            if r.label in labels:
                raise ValueError("Input rule cannot have same label, '{0}', as any other rule.".format(r.label))
            labels.append(r.label)
        self.graph = nx.compose(self.graph, rule.graph)
        try:
            self.add_rule_n(rule)
        except:
            pass

    def add_rule_n(self, rule):
        graph, color = rule.graph_n
        U = nx.Graph()
        if 'graph_n' in dir(self):
            U.add_edges_from(self.graph_n[0].edges())
            U.add_nodes_from(self.graph_n[0].nodes())
        U.add_edges_from(graph.edges())
        U.add_nodes_from(graph.nodes())
        if not 'colors' in dir(self):
            self.colors = []
            self.colors.extend(color)
        else:
            self.colors.extend(color)
        self.graph_n = U, self.colors

    def view(self):
        fig, ax = ControlSystemVisualizer(self).view()
        fig.show()

    def view_n(self):
        fig, ax = ControlSystemVisualizer(self).view_n()
        fig.show()

class ControlSystemSimulation(object):
    def __init__(self, control_system, clip_to_bounds=True, cache=True, flush_after_run=1000):
        assert isinstance(control_system, ControlSystem)
        self.ctrl = control_system
        self.input = _InputAcceptor(self)
        self.output = OrderedDict()
        self.cache = cache
        self._array_inputs = False
        self._array_shape = None
        self.unique_id = self._update_unique_id()
        self.clip_to_bounds = clip_to_bounds
        self._calculated = []
        self._run = 0
        self._flush_after_run = flush_after_run

    def _update_unique_id(self):
        if not self._array_inputs:
            self.unique_id = (str(id(self.ctrl)) + str(hash(self._get_inputs().__repr__())))

    def _get_inputs(self):
        return self.input._get_inputs()

    def inputs(self, input_dict):
        for label, value in input_dict.items():
            self.input[label] = value

    def compute(self):
        self.input._update_to_current()
        if self._array_inputs:
            self.cache = False
            self._clear_outputs()
        if self.cache is not False and self.unique_id in self._calculated:
            for consequent in self.ctrl.consequents:
                self.output[consequent.label] = consequent.output[self]
            return
        for antecedent in self.ctrl.antecedents:
            if antecedent.input[self] is None:
                raise ValueError("All antecedents must have input values!")
            CrispValueCalculator(antecedent, self).fuzz(antecedent.input[self])
        first = True
        for rule in self.ctrl.rules:
            if first:
                for c in rule.consequent:
                    c.term.membership_value[self] = None
                    c.activation[self] = None
                first = False
            self.compute_rule(rule)
        for consequent in self.ctrl.consequents:
            consequent.output[self] = CrispValueCalculator(consequent, self).defuzz()
            self.output[consequent.label] = consequent.output[self]
        if self.cache is not False:
            self._calculated.append(self.unique_id)
        else:
            self._reset_simulation()
        self._run += 1
        if self._run % self._flush_after_run == 0:
            self._reset_simulation()

    def compute_rule(self, rule):
        if isinstance(rule.antecedent, TermAggregate):
            rule.antecedent.agg_methods = rule._aggregation_methods
        rule.aggregate_firing[self] = rule.antecedent.membership_value[self]
        for c in rule.consequent:
            assert isinstance(c, WeightedTerm)
            c.activation[self] = rule.aggregate_firing[self] * c.weight
        for c in rule.consequent:
            assert isinstance(c, WeightedTerm)
            term = c.term
            value = c.activation[self]
            if term.membership_value[self] is None:
                term.membership_value[self] = value
            else:
                accu = term.parent.accumulation_method
                term.membership_value[self] = accu(value, term.membership_value[self])
            term.cuts[self][rule.label] = term.membership_value[self]

    def reset(self):
        self._reset_simulation()

    def _reset_simulation(self):
        def _clear_terms(fuzzy_var):
            for term in fuzzy_var.terms.values():
                term.membership_value.clear()
                term.cuts.clear()
        for rule in self.ctrl.rules:
            rule.aggregate_firing.clear()
            for c in rule.consequent:
                c.activation.clear()
        for consequent in self.ctrl.consequents:
            consequent.output.clear()
            _clear_terms(consequent)
        for antecedent in self.ctrl.antecedents:
            antecedent.input.clear()
            _clear_terms(antecedent)
        self._calculated = []
        self._run = 0

    def _clear_outputs(self):
        def _clear_terms(fuzzy_var):
            for term in fuzzy_var.terms.values():
                term.membership_value.clear()
                term.cuts.clear()
        for rule in self.ctrl.rules:
            rule.aggregate_firing.clear()
            for c in rule.consequent:
                c.activation.clear()
        for consequent in self.ctrl.consequents:
            consequent.output.clear()
            _clear_terms(consequent)
        self._calculated = []
        self._run = 0

    def print_state(self):
        if next(self.ctrl.consequents).output[self] is None:
            raise ValueError("Call compute method first.")
        print("=============")
        print(" Antecedents ")
        print("=============")
        for v in self.ctrl.antecedents:
            print("{0:<35} = {1}".format(v, v.input[self]))
            for term in v.terms.values():
                print("  - {0:<32}: {1}".format(term.label, term.membership_value[self]))
        print("")
        print("=======")
        print(" Rules ")
        print("=======")
        rule_number = {}
        for rn, r in enumerate(self.ctrl.rules):
            assert isinstance(r, Rule)
            rule_number[r] = "RULE #%d" % rn
            print("RULE #%d:\n  %s\n" % (rn, r))
            print("  Aggregation (IF-clause):")
            for term in r.antecedent_terms:
                assert isinstance(term, Term)
                print("  - {0:<55}: {1}".format(term.full_label, term.membership_value[self]))
            print("    {0:>54} = {1}".format(r.antecedent, r.aggregate_firing[self]))
            print("  Activation (THEN-clause):")
            for c in r.consequent:
                assert isinstance(c, WeightedTerm)
                print("    {0:>54} : {1}".format(c, c.activation[self]))
            print("")
        print("==============================")
        print(" Intermediaries and Consequents ")
        print("==============================")
        for c in self.ctrl.consequents:
            print("{0:<36} = {1}".format(c, CrispValueCalculator(c, self).defuzz()))
            for term in c.terms.values():
                print("  %s:" % term.label)
                for cut_rule, cut_value in term.cuts[self].items():
                    if cut_rule not in rule_number.keys():
                        continue
                    print("    {0:>32} : {1}".format(rule_number[cut_rule], cut_value))
                accu = "Accumulate using %s" % c.accumulation_method.func_name
                print("    {0:>32} : {1}".format(accu, term.membership_value[self]))
            print("")

class CrispValueCalculator(object):
    def __init__(self, fuzzy_var, sim):
        assert isinstance(fuzzy_var, FuzzyVariable)
        assert isinstance(sim, ControlSystemSimulation)
        self.var = fuzzy_var
        self.sim = sim

    def defuzz(self):
        if not self.sim._array_inputs:
            ups_universe, output_mf, cut_mfs = self.find_memberships()
            if len(cut_mfs) == 0:
                raise ValueError("No terms have memberships. Make sure you have at least one rule connected to this variable and have run the rules calculation.")
            try:
                return defuzz(ups_universe, output_mf, self.var.defuzzify_method)
            except AssertionError:
                raise ValueError("Crisp output cannot be calculated, likely because the system is too sparse. Check to make sure this set of input values will activate at least one connected Term in each Antecedent via the current set of Rules.")
        else:
            output = cp.zeros(self.sim._array_shape, dtype=cp.float64)
            it = cp.nditer(output, ['multi_index'], [['writeonly', 'allocate']])
            for out in it:
                universe, mf = self.find_memberships_nd(it.multi_index)
                out[...] = defuzz(universe, mf, self.var.defuzzify_method)
            return output

    def fuzz(self, value):
        if len(self.var.terms) == 0:
            raise ValueError("Set Term membership function(s) first")
        for label, term in self.var.terms.items():
            term.membership_value[self.sim] = interp_membership(self.var.universe, term.mf, value)

    def find_memberships(self):
        new_values = []
        for label, term in self.var.terms.items():
            term._cut = term.membership_value[self.sim]
            if term._cut is None:
                continue
            new_values.extend(_interp_universe_fast(self.var.universe, term.mf, term._cut).tolist())
        new_universe = cp.union1d(self.var.universe, new_values)
        output_mf = cp.zeros_like(new_universe, dtype=cp.float64)
        term_mfs = {}
        for label, term in self.var.terms.items():
            if term._cut is None:
                continue
            upsampled_mf = interp_membership(self.var.universe, term.mf, new_universe)
            term_mfs[label] = cp.minimum(term._cut, upsampled_mf)
            cp.maximum(output_mf, term_mfs[label], output_mf)
        return new_universe, output_mf, term_mfs

    def find_memberships_nd(self, idx):
        new_values = []
        for label, term in self.var.terms.items():
            term._cut = term.membership_value[self.sim][idx]
            if term._cut is None:
                continue
            new_values.extend(_interp_universe_fast(self.var.universe, term.mf, term._cut).tolist())
        new_universe = cp.union1d(self.var.universe, new_values)
        output_mf = cp.zeros_like(new_universe, dtype=cp.float64)
        term_mfs = {}
        for label, term in self.var.terms.items():
            if term._cut is None:
                continue
            upsampled_mf = interp_membership(self.var.universe, term.mf, new_universe)
            term_mfs[label] = cp.minimum(term._cut, upsampled_mf)
            cp.maximum(output_mf, term_mfs[label], output_mf)
        return new_universe, output_mf

class RuleOrderGenerator(object):
    def __init__(self, control_system):
        assert isinstance(control_system, ControlSystem)
        self.control_system = control_system
        self._cache = []
        self._cached_graph = None

    def __iter__(self):
        if self._cached_graph is not self.control_system.graph:
            self._init_state()
            self._cache = list(self._process_rules(self.all_rules[:]))
            self._cached_graph = self.control_system.graph
        for n, r in enumerate(self._cache):
            yield r
        else:
            n = 0
        if n == 0:
            pass
        else:
            assert n == len(self.all_rules) - 1, "Not all rules exposed"

    def _init_state(self):
        self.calced_graph = nx.DiGraph()
        for a in self.control_system.antecedents:
            for t in a.terms.values():
                self.calced_graph.add_edge(a, t)
        self.all_graph = self.control_system.graph
        self.all_rules = []
        for node in self.all_graph.nodes():
            if isinstance(node, Rule):
                self.all_rules.append(node)

    def _process_rules(self, rules):
        len_rules = len(rules)
        skipped_rules = []
        while len(rules) > 0:
            rule = rules.pop(0)
            if self._can_calc_rule(rule):
                yield rule
                self.calced_graph = nx.compose(self.calced_graph, rule.graph)
            else:
                skipped_rules.append(rule)
        if len(skipped_rules) == 0:
            try:
                return
            except StopIteration:
                return
        else:
            if len(skipped_rules) == len_rules:
                raise RuntimeError("Unable to resolve rule execution order. The most likely reason is two or more rules that depend on each other.\nPlease check the rule graph for loops.")
            else:
                for r in self._process_rules(skipped_rules):
                    yield r

    def _can_calc_rule(self, rule):
        try:
            predecessors = self.all_graph.predecessors(rule)
        except AttributeError:
            predecessors = self.all_graph.predecessors(rule)
        for p in predecessors:
            assert isinstance(p, Term)
            if p not in self.calced_graph:
                return False
            try:
                all_degree = len(self.all_graph.predecessors(p))
                calced_degree = len(self.calced_graph.predecessors(p))
            except TypeError:
                all_degree = self.all_graph.predecessors(p).__sizeof__()
                calced_degree = self.calced_graph.predecessors(p).__sizeof__()
            if all_degree != calced_degree:
                return False
        return True
