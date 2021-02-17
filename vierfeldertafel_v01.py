from sympy import *

import pandas as pd

import graphviz as gv

class TreeItem():
    def __init__(self, name, value, subtree=[]):
        self.name = name
        self.value = value
        self.subtree = subtree
                                        
    def create_graph(self,g=None, digits=None):
        if g is None: # startpunkt des Wahrscheinlichkeitsbaumes
            g = gv.Graph(comment = self.name)
            self.nodename = self.name
            g.node(self.nodename,shape='point')
            self.res = S(1) # der Startpunkt hat die Wahrscheinlichkeit 1
            
        # process nodes and edges:
        for s in self.subtree:
            # eindeutigen Namen für jeden node vergeben
            s.nodename = f'{self.nodename}.{s.name}'
            
            # resultierende Wahrscheinlichkeit dieses nodes
            s.res = self.res * s.value 
            
            # node anlegen
            g.node(s.nodename, label=s.name, shape='circle')
            
            # Verbindung von parent zu node, label ist bedingte Wahrscheinlichkeit
            label = str(s.value.n(digits)) if digits else str(s.value)
            g.edge(self.nodename, s.nodename, label=label)
            
        # process subtrees:
        # g rekursiv erweitern
        if self.subtree:
            for s in self.subtree:
                g = s.create_graph(g,digits)
        else:
            # am Ende des Baumes angelangt
            # Es wird noch ein Knoten eingefügt, der die 
            # Wahrscheinlichkeit dieses Pfades angibt
            nodename = f'{self.nodename}.res'
            #label = str(self.res.n(digits)) if digits else str(self.res)
            label = f'{self.res.n(digits) if digits else self.res}'
            g.node(nodename, label=label,shape='box')
            g.edge(self.nodename,nodename,style='dotted')
        
        return g

class Vierfelder_Tafel():
    ''' In der Vierfeldertafel treten zwei Ereignisse 
    
            A, B
    
        und die zugehörigen Gegenereignisse
        
            Ā, B̄
            
        auf. Diesen Ereignissen sind Grundwahrscheinlichkeiten
        
            a1=P(A), a2=P(Ā)
            b1=P(B), b2=P(B̄)
            
        zugeordnet, die sich zu 1 addieren müssen. 
        
            a1+a2 = 1
            b1+b2 = 1
        
        In der Vierfeldertafel treten zusätzlich 
        Schnittwahrscheinlichkeiten
        
            a1b1=P(A∩B)=b1a1, a2b1=P(Ā∩B)=b1a2
            a1b2=P(A∩B̄)=b2a1, a2b2=P(Ā∩B̄)=b2a2
            
        auf, die symmetrisch sind. Sie erfüllen die Gleichungen
        
            a1b1 + a2b1 = b1
            a1b2 + a2b2 = b2
            
            b1a1 + b2a1 = a1
            b1a2 + b2a2 = a2
        
        Weiter gibt es die bedingten 
        Wahrscheinlichkeiten, die durch
        
            a1_b1=P(A|B)=P_B(A), a2_b1=P(Ā|B)=P_B(Ā)
            a1_b2=P(A|B̄)=P_B̄(A), a2_b2=P(Ā|B̄)=P_B̄(Ā)
            
        definiert sind, auf. Für sie gelten die Gleichungen
        
            a1_b1*b1 = a1b1, a2_b1*b1 = a2b1
            a1_b2*b2 = a1b1, a2_b2*b2 = a2b2
            
        sowie
        
            b1_a1*a1 = b1a1, b2_a1*a1 = b2a1
            b1_a2*a2 = b1a2, b2_a2*a2 = b2a2
        
        Um eine Vierfeldertafel eindeutig festzulegen, müssen
        drei unabhängige Größen dieser Wahrscheinlichkeiten
        bekannt sein. 
        
        Es ist möglich, widersprüchliche Bedingungen zu
        formulieren. Stets müssen alle Wahrscheinlichkeiten p
        der Vierfeldertafel die Forderung 
        
            0 <= p <= 1
        
        erfüllen.
    '''
    # Grundwahrscheinlichkeiten
    # P(A) = a1, P(Ā) = a2, u.s.w.
    a1,a2,b1,b2 = symbols('{a1},{a2},{b1},{b2}')
    
    # Schnittwahrscheinlichkeiten
    # P(A∩B) = a1b1, u.s.w.
    a1b1, a1b2, a2b1, a2b2 = symbols('a1b1,a1b2,a2b1,a2b2')
    b1a1, b2a1, b1a2, b2a2 = symbols('a1b1,a1b2,a2b1,a2b2') # Symmetrie!
    
    # Bedingte Wahrscheinlichkeiten
    # P(A|B) = a1_b1, P(A|B̄) = a1_b2, u.s.w.
    a1_b1, a1_b2, a2_b1, a2_b2 = symbols('{a1}_{b1},{a1}_{b2},{a2}_{b1},{a2}_{b2}')
    b1_a1, b1_a2, b2_a1, b2_a2 = symbols('{b1}_{a1},{b1}_{a2},{b2}_{a1},{b2}_{a2}')

    
    # Gleichungen
    base_eqns = [
        Eq(a1b1+a1b2,a1),  # Vierfeldertafel 1-te Zeile
        Eq(a2b1+a2b2,a2),  # Vierfeldertafel 2-te Zeile
        Eq(b1a1+b1a2,b1),  # Vierfeldertafel 1-te Spalte
        Eq(b2a1+b2a2,b2),  # Vierfeldertafel 2-te Spalte
        Eq(b1+b2,1),       # Vierfeldertafel 3-te Spalte
        Eq(a1_b1*b1,a1b1), # Bedingte Wahrscheinlichkeit P(A1|B1)
        Eq(a1_b2*b2,a1b2), # Bedingte Wahrscheinlichkeit P(A1|B2)
        Eq(a2_b1*b1,a2b1), # Bedingte Wahrscheinlichkeit P(A2|B1)
        Eq(a2_b2*b2,a2b2), # Bedingte Wahrscheinlichkeit P(A2|B1)
        Eq(b1_a1*a1,b1a1), # Bedingte Wahrscheinlichkeit P(B1|A1)
        Eq(b1_a2*a2,b1a2), # Bedingte Wahrscheinlichkeit P(B1|A2)
        Eq(b2_a1*a1,b2a1), # Bedingte Wahrscheinlichkeit P(B2|A1)
        Eq(b2_a2*a2,b2a2), # Bedingte Wahrscheinlichkeit P(B2|A2)
    ]
    
    def __init__(self, **kwargs):
        ''' Die Angabe von drei Wahrscheinlichkeiten der Form
            
            Vierfeldertafel(a1 = 0.3, a1b1 = 0.15, b1 = 0.5)
            
            führen auf eine vollständig bestimmte Vierfeldertafel.
            
            Alle Wahrscheinlichkeiten müssen zwischen 0 und 1 liegen:
            
                0 < a1,a1b1,b1 < 1
                
            Zusätzlich müssen alle Wahrscheinlichkeiten der 
            berechneten Vierfeldertafel zwischen 0 und 1 liegen.
        '''
        
        # die gegebenen Wahrscheinlichkeiten als Gleichungen formulieren
        eqns = [Eq(getattr(self,k),v) for k,v in kwargs.items()]
        
        # das Gleichungssystem der Vierfeldertafel lösen
        self.lsg = solve(eqns + self.base_eqns)

    @property
    def anzahl_loesungen(self):
        return len(self.lsg)
    
    def tafel(self,loesung_idx=0, digits=None):
        i = loesung_idx
        df =  pd.DataFrame(
            [
                [self.lsg[i][getattr(self,k)] for k in ['b1a1','b1a2','b1']],
                [self.lsg[i][getattr(self,k)] for k in ['b2a1','b2a2','b2']],
                [self.lsg[i][getattr(self,k)] for k in ['a1','a2']] + [S(1)],
            ],
            columns=[r'$A$',r'$\bar{A}$', r'$\sum$'],
            index=[r'$B$',r'$\bar{B}$', r'$\sum$']
        )
        
        if digits:
            for col in df.columns:
                df[col] = df[col].apply(lambda x: x.n(digits))

        return df
    
    def tree_a(self,loesung_idx=0,digits=None):
        i = loesung_idx
        t = TreeItem(
            'O',S(1),
            [
                TreeItem(
                    'A',self.lsg[i][self.a1],
                    [
                        TreeItem('B',self.lsg[i][self.b1_a1]),
                        TreeItem('B̄',self.lsg[i][self.b2_a1]),
                    ]
                ),
                TreeItem(
                    'Ā',self.lsg[i][self.a2],
                    [
                        TreeItem('B',self.lsg[i][self.b1_a2]),
                        TreeItem('B̄',self.lsg[i][self.b2_a2]),
                    ]
                ),
            ]
        )
        return t.create_graph(digits=digits)
        
    def tree_b(self,loesung_idx=0,digits=None):
        i = loesung_idx
        t = TreeItem(
            'O',S(1),
            [
                TreeItem(
                    'B',self.lsg[i][self.b1],
                    [
                        TreeItem('A',self.lsg[i][self.a1_b1]),
                        TreeItem('Ā',self.lsg[i][self.a2_b1]),
                    ]
                ),
                TreeItem(
                    'B̄',self.lsg[i][self.b2],
                    [
                        TreeItem('A',self.lsg[i][self.a1_b2]),
                        TreeItem('Ā',self.lsg[i][self.a2_b2]),
                    ]
                ),
            ]
        )
        return t.create_graph(digits=digits)
    
    def get_value(self,key,loesung_idx=0,digits=None):
        i = loesung_idx
        value = self.lsg[i][key]
        return value.n(digits) if digits else value


