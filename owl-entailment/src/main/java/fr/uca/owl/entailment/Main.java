package fr.uca.owl.entailment;

import org.semanticweb.owlapi.model.*;

import java.util.HashSet;
import java.util.Set;


public class Main {
    public static void main(String[] args){


        OWLOntology axioms = Utils.checkOntology("./dbpedia-axioms sandbox.owl");
        Set<OWLAxiom> axiomSet = new HashSet<OWLAxiom>();
        axiomSet.addAll(axioms.getLogicalAxioms());

        OWLOntology ontology = Utils.checkOntology("./protoge-test.owl");
        //OWLOntology ontology = Utils.checkOntology("./protoge-test.owl");
        //OWLOntology ontology = Utils.checkOntology("./dbpedia_3.9.owl");
        //OWLOntology ontology = Utils.checkOntology("./dbpedia_2015-04.owl");
        long start = System.currentTimeMillis();

        EntailmentCheck checker = new EntailmentCheck();
        checker.checkEntailment(ontology,axiomSet);

        long end = System.currentTimeMillis();
        System.out.println("Time = " + (end-start) + "ms");

    }

}
