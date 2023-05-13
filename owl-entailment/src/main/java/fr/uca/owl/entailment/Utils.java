package fr.uca.owl.entailment;


import java.io.File;

import org.semanticweb.HermiT.Reasoner;
import org.semanticweb.owlapi.apibinding.OWLManager;
import org.semanticweb.owlapi.model.OWLOntology;
import org.semanticweb.owlapi.model.OWLOntologyCreationException;
import org.semanticweb.owlapi.model.OWLOntologyManager;
import org.semanticweb.owlapi.reasoner.OWLReasoner;

/**
 *
 * @author Iliana Petrova
 */
public class Utils {

    static OWLOntology checkOntology(String fileName){
        
        OWLOntology ontology = null;
        OWLOntologyManager manager = OWLManager.createOWLOntologyManager();
        try{
            ontology = manager.loadOntologyFromOntologyDocument(new File(fileName));
        }
        catch(OWLOntologyCreationException ex){
            System.out.println("Error loading the ontology.");
        }
        
        return ontology;
    }

    static OWLReasoner createReasoner(OWLOntology ontology){

        OWLReasoner reasoner = null;

        try{

            //HermiT
            Reasoner.ReasonerFactory reasonerFactory = new Reasoner.ReasonerFactory();
            reasoner = reasonerFactory.createReasoner(ontology);

        }
        catch(Exception ex){
            System.out.println("Error init reasoner.");
        }

        return reasoner;
    }
    

}