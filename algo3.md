
### **1. Tahmin Aşamasında Ağaçların Bağımsızlığı**

**Bağımsızlık Varsayımı:**

- Her ağaç \( t \in \{1, 2, ..., T\} \) tahmin aşamasında bağımsız olarak çalışır.
- Her ağaç \( f_t(x) \) şeklinde tahminler üretir.
- Toplam model tahmini, tüm ağaçların tahminlerinin toplamıdır:

  \[
  F(x) = \sum_{t=1}^{T} f_t(x)
  \]

Bu varsayım, tahmin sürecinde ağaçların birbirinden bağımsız olarak katkı yaptığını ve bu nedenle her bir ağacın ayrı ayrı analiz edilebileceğini ifade eder.

### **2. Özelliğin Split Node Olarak Kullanıldığı Ağaçların Seçimi**

- **Özelliğin Kullanıldığı Ağaçlar Kümesi:**

  \[
  S_x = \{ t \mid \text{Özellik } x, \text{ ağacı } t \text{'de split node olarak kullanılmıştır} \}
  \]

- **Toplam ağaç sayısı:** \( T \)
- **x özelliğini kullanan ağaç sayısı:** \( |S_x| = m \leq T \)

Bu adımda, x özelliğinin etkisini analiz etmek için sadece x'i split node olarak kullanan ağaçları dikkate alıyoruz.

### **3. Her Split Node İçin Yaprak İstatistikleri**

- **Ağaç \( t \) ve dal \( i \) için:**

  - \( L_{t,i} \): Ağaç \( t \)'deki dal \( i \)'nin yaprak sayısı (leaf count)
  - \( V_{t,i} \): Ağaç \( t \)'deki dal \( i \)'nin yaprak değeri (leaf value)
  - \( D_t \): Ağaç \( t \)'de x özelliğinin kullanıldığı dalların kümesi

Bu bilgiler, her ağacın x özelliği için nasıl tahminler yaptığını ve bu tahminlerin ne kadar güçlü olduğunu gösterir.

### **4. Beklenen Değerin Hesaplanması**

**Her Ağaç İçin Beklenen Değer:**

- x özelliğinin belirli bir aralığı için (örneğin, \( x \in \text{interval} \)), ağaç \( t \)'nin beklenen değeri:

  \[
  E[f_t(x) \mid x \in \text{interval}] = \frac{\sum\limits_{i \in D_t} L_{t,i} \cdot V_{t,i}}{\sum\limits_{i \in D_t} L_{t,i}}
  \]

  - Bu formül, yaprak değerlerinin yaprak sayılarıyla ağırlıklandırılmış ortalamasıdır.

**Her Ağaç için ortalama veri sayısı:**

  \[
  AC[x \in \text{interval}] = \frac{1}{|S_x|} \sum_{t \in S_x} \left( \sum_{i \in D_t} L_{t,i} \right)
  \]

  - **Bağımsızlık Varsayımı:** Ağaçlar tahmin aşamasında bağımsız olduğundan, her ağacın katkısı eşit ağırlıklıdır.
  - **Yaprak Sayıları:** Yaprak sayıları, o yaprağa düşen örneklerin sayısını gösterir ancak bağımsız oldukları için bir ağırlık belirtmezler.
  - **Büyük Sayılar Yasası:** Yeterli sayıda ağaç olduğunda, ortalama yaprak sayıları gerçek dağılıma yakınsar.



**Genel Beklenen Değer:**

- Tüm x'i kullanan ağaçlar için beklenen değerin ortalaması:

\[
E[F(x)] = \frac{\sum_{t \in S_x} E[f_t(x) \mid x \in \text{interval}] \cdot AC[x \in \text{interval}]}{\sum_{t \in S_x} AC[x \in \text{interval}]}
\]

  - Bu, tüm ilgili ağaçların intervallarının katkılarının ortalamasını alır.
  

### **6. Yaklaşımın Matematiksel Gerekçelendirilmesi**

#### **a. Gürültünün Azaltılması (Noise Reduction)**

- **x özelliğini kullanmayan ağaçlar (\( t \notin S_x \)) ve dallar (\( i \in D_t \)) hesaba katılmıyor çünkü:**

  - Bu ağaçlarda ve dallarda x'in etkisi doğrudan gözlemlenemez.

  - Bu nedenle, x'in marjinal etkisini ölçerken bu ağaçları ve dalları dahil etmek, sonuçlara gürültü ekler.


#### **b. Beklenen Değerin Anlamlılığı**

- **Marjinal Etki Ölçümü:**

  \[
  \text{Contribution}(x \in \text{interval}) = E[F(x) \mid x \in \text{interval}] - E[F(x)]
  \]

  - Bu fark, x'in belirli bir aralıktaki model tahminine olan katkısını gösterir.


#### **d. ????**

- **Tutarlılık (Consistency):**

  - **n (veri sayısı) ve T (ağaç sayısı) sonsuza giderken:**

    - Ortalama yaprak sayıları, gerçek veri dağılımına yakınsar.
    - Beklenen değerler, gerçek koşullu beklenen değerlere yakınsar.

- **Sonuç:**

  - Yaklaşım, yeterli veri ve ağaç sayısıyla, x'in model üzerindeki etkisini yansıtacak şekilde tasarlanmıştır.
