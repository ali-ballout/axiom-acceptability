package fr.uca.owl.entailment;

import java.util.ArrayList;
import java.util.Map;
import java.util.Set;

import org.semanticweb.owlapi.apibinding.OWLManager;
import org.semanticweb.owlapi.model.*;
import org.semanticweb.owlapi.reasoner.OWLReasoner;

/**
 *
 * @author Iliana Petrova
 */
public class EntailmentCheck {

    public void checkEntailment(OWLOntology ontology, Set<OWLAxiom> axiomSet){

        OWLReasoner reasoner = Utils.createReasoner(ontology);
        int count = 0;
        for (OWLAxiom ax: axiomSet) {
            if(!reasoner.isEntailed(ax)){
                //System.out.println("Axiom: " + ax + "not entailed.");
                System.out.println(ax);
                count++;
            }
        }
    //print the number of axioms that are not entailed
    System.out.println("Number of axioms not entailed: " + count);
    }


    public EntailmentCheck(){
    }
}